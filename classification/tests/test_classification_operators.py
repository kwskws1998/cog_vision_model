"""Independent numerical checks for the proposal and the pinned external loss."""

import math
import unittest
from dataclasses import replace

import numpy as np
import torch
from clvision.config import Config, recipe
from clvision.losses import consistency_loss, training_objective, zipf_distribution, zipf_kl
from clvision.models import SmallResNet18
from torch.nn import functional as F

from cue_routing.io import seed_everything
from cue_routing.models import CueReadout
from cue_routing.routing import route_gates, routed_loss


def numpy_probabilities(logits):
    """Calculate the reference categorical probabilities without using either routing operator."""
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def numpy_routed_gradient(logits, areas, targets, gates=None):
    """Apply the closed-form a_k/B [gamma_k p_joint + (1-gamma_k) p_k - y] derivative."""
    joint = numpy_probabilities((logits * areas[None, :, None]).sum(axis=1))
    local = numpy_probabilities(logits)
    if gates is None:
        discrepancy = np.linalg.norm(local - joint[:, None], axis=-1) / np.sqrt(2)
        gates = 0.5 + 0.5 / (1 + np.exp(-(discrepancy - 0.15) / 0.05))
    one_hot = np.eye(logits.shape[-1])[targets, None]
    gradient = (
        areas[None, :, None]
        / len(targets)
        * (gates[..., None] * joint[:, None] + (1 - gates[..., None]) * local - one_hot)
    )
    global_ce = -np.log(joint[np.arange(len(targets)), targets]).mean()
    kl = (joint[:, None] * (np.log(joint[:, None]) - np.log(local))).sum(axis=-1)
    objective = global_ce + (areas[None] * (1 - gates) * kl).sum(axis=1).mean()
    return gradient, objective


class ClassificationOperatorTests(unittest.TestCase):
    """Compare complete gradients and explicit independent reference calculations."""

    def test_gradient_equivalence_all_gates_and_class_counts(self):
        """The KL rewrite must match the original routed gradient, including shared heads."""
        torch.manual_seed(37)
        for classes in (2, 7, 100, 200):
            for method in (
                "cue",
                "fixed_global_local",
                "pure_marginal",
                "image_mean_gate",
                "permuted_gate",
            ):
                values = torch.randn(3, 4, 11, dtype=torch.float64, requires_grad=True)
                matrix = torch.randn(11, classes, dtype=torch.float64, requires_grad=True)
                bias = torch.randn(classes, dtype=torch.float64, requires_grad=True)
                local = values @ matrix + bias
                areas = torch.tensor([9, 12, 12, 16], dtype=torch.float64) / 49
                targets = torch.arange(3) % classes
                config = replace(Config(), method=method)
                rewritten = consistency_loss(local, areas, targets, config)[0]
                gates, _, _ = route_gates(local, areas)
                if method == "fixed_global_local":
                    gates = torch.full_like(gates, 0.75)
                elif method == "pure_marginal":
                    gates = torch.zeros_like(gates)
                elif method == "image_mean_gate":
                    gates = (gates * areas[None]).sum(1, keepdim=True).expand_as(gates)
                elif method == "permuted_gate":
                    gates = gates[:, [2, 3, 0, 1]]
                original = routed_loss(local, targets, areas, gates_override=gates)[0]
                left = torch.autograd.grad(rewritten, (values, matrix, bias), retain_graph=True)
                right = torch.autograd.grad(original, (values, matrix, bias))
                for actual, expected in zip(left, right):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_consistency_endpoints(self):
        """Zero consistency recovers CE; full consistency has pure marginal gradients."""
        local = torch.randn(5, 4, 7, dtype=torch.float64, requires_grad=True)
        targets = torch.arange(5)
        areas = torch.full((4,), 0.25, dtype=torch.float64)
        zero = consistency_loss(
            local, areas, targets, replace(Config(), method="fixed_global_local", fixed_lambda=0)
        )[0]
        global_ce = F.cross_entropy(local.mean(1), targets)
        torch.testing.assert_close(
            torch.autograd.grad(zero, local, retain_graph=True)[0],
            torch.autograd.grad(global_ce, local, retain_graph=True)[0],
        )
        marginal = consistency_loss(
            local, areas, targets, replace(Config(), method="pure_marginal")
        )[0]
        independent = F.cross_entropy(
            local.reshape(-1, 7), targets[:, None].expand(5, 4).reshape(-1)
        )
        torch.testing.assert_close(
            torch.autograd.grad(marginal, local, retain_graph=True)[0],
            torch.autograd.grad(independent, local)[0],
        )

    def test_cue_closed_form_gradient_and_scalar_with_unequal_areas(self):
        """Check the detached teacher/gate derivative and both endpoints against independent NumPy."""
        generator = np.random.default_rng(9021)
        areas = np.array([9, 12, 12, 16], dtype=np.float64) / 49
        for classes in (2, 7, 100, 200):
            values = generator.normal(0, 3, size=(5, 4, classes))
            targets = generator.integers(classes, size=5)
            for method, fixed_lambda, gates in (
                ("cue", 0.25, None),
                ("fixed_global_local", 0, np.ones((5, 4))),
                ("pure_marginal", 0.25, np.zeros((5, 4))),
            ):
                with self.subTest(classes=classes, method=method):
                    local = torch.tensor(values, requires_grad=True)
                    config = replace(Config(), method=method, fixed_lambda=fixed_lambda)
                    loss, _, _ = consistency_loss(
                        local, torch.tensor(areas), torch.tensor(targets), config
                    )
                    expected_gradient, expected_loss = numpy_routed_gradient(
                        values, areas, targets, gates
                    )
                    np.testing.assert_allclose(
                        torch.autograd.grad(loss, local)[0].numpy(),
                        expected_gradient,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    self.assertAlmostEqual(loss.detach().item(), expected_loss, places=12)

    def test_cue_closed_form_shared_spatial_head_derivatives(self):
        """Propagate the analytic logit derivative through unequal regions and the shared affine head."""
        generator = np.random.default_rng(9022)
        readout = CueReadout(3, 7).double()
        with torch.no_grad():
            readout.classifier.weight.copy_(torch.tensor(generator.normal(size=(7, 3))))
            readout.classifier.bias.copy_(torch.tensor(generator.normal(size=7)))
        values = generator.normal(size=(5, 3, 7, 7))
        targets = generator.integers(7, size=5)
        features = torch.tensor(values, requires_grad=True)
        local, areas = readout.cue_logits(features)
        loss, joint, _ = consistency_loss(local, areas, torch.tensor(targets), Config())
        actual = torch.autograd.grad(
            loss, (features, readout.classifier.weight, readout.classifier.bias)
        )
        regions = ((0, 3, 0, 3), (0, 3, 3, 7), (3, 7, 0, 3), (3, 7, 3, 7))
        pooled = np.stack(
            [values[:, :, y0:y1, x0:x1].mean(axis=(2, 3)) for y0, y1, x0, x1 in regions],
            axis=1,
        )
        weights = readout.classifier.weight.detach().numpy()
        bias = readout.classifier.bias.detach().numpy()
        reference_areas = np.array([9, 12, 12, 16], dtype=np.float64) / 49
        logits = pooled @ weights.T + bias
        derivative, _ = numpy_routed_gradient(logits, reference_areas, targets)
        feature_gradient = np.zeros_like(values)
        for region, (y0, y1, x0, x1) in enumerate(regions):
            feature_gradient[:, :, y0:y1, x0:x1] = (derivative[:, region] @ weights)[
                :, :, None, None
            ] / ((y1 - y0) * (x1 - x0))
        expected = (
            feature_gradient,
            np.einsum("bkc,bkf->cf", derivative, pooled),
            derivative.sum(axis=(0, 1)),
        )
        for observed, reference in zip(actual, expected):
            np.testing.assert_allclose(observed.numpy(), reference, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(areas.numpy(), reference_areas, rtol=0, atol=0)
        np.testing.assert_allclose(local.detach().numpy(), logits, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            joint.detach().numpy(),
            values.mean(axis=(2, 3)) @ weights.T + bias,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zipf_minimum_ties_and_zero_vote_classes(self):
        """Use an independent Python rank count and normalized probability reference."""
        votes = torch.tensor([[0, 0, 1, 1, 4], [3, 3, 3, 3, 3], [0, 1, 2, 3, 4]])
        targets = torch.tensor([4, 3, 0])
        reference = []
        for row, target in zip(votes.tolist(), targets.tolist()):
            counts = [row.count(label) for label in range(5)]
            weights = [1 / (1 + sum(other > count for other in counts)) for count in counts]
            weights = [value for label, value in enumerate(weights) if label != target]
            reference.append([value / sum(weights) for value in weights])
        torch.testing.assert_close(
            zipf_distribution(votes, targets, 5), torch.tensor(reference, dtype=torch.float64)
        )

    def test_zipf_value_gradient_and_target_exclusion(self):
        """Check KL against NumPy and a finite difference independent of torch autograd."""
        logits = torch.tensor([[1.3, -0.5, 2.1, 0.7]], dtype=torch.float64, requires_grad=True)
        targets, votes = torch.tensor([2]), torch.tensor([[0, 1, 0, 3]])
        loss = zipf_kl(logits, votes, targets)
        gradient = torch.autograd.grad(loss, logits)[0]
        q = np.array([0.5, 0.25, 0.25])

        def reference(values):
            selected = values[[0, 1, 3]]
            p = np.exp(selected - selected.max())
            p /= p.sum()
            return float(np.sum(q * (np.log(q) - np.log(p))))

        values = logits.detach().numpy()[0]
        self.assertAlmostEqual(float(loss), reference(values), places=12)
        for index in range(4):
            delta = np.zeros(4)
            delta[index] = 1e-6
            numerical = (reference(values + delta) - reference(values - delta)) / 2e-6
            self.assertAlmostEqual(float(gradient[0, index]), numerical, places=8)
        self.assertEqual(float(gradient[0, 2]), 0.0)
        torch.testing.assert_close(loss, zipf_kl(logits + 731, votes, targets))

    def test_zipf_binary_non_target_loss_is_zero(self):
        """A single non-target category has no artificial distillation entropy."""
        logits = torch.randn(4, 2, requires_grad=True)
        loss = zipf_kl(logits, torch.zeros(4, 12, dtype=torch.long), torch.tensor([0, 1, 0, 1]))
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits)))

    def test_shared_initialization_and_rng(self):
        """The Zipf auxiliary head must not change common initial tensors or training RNG."""
        hashes, random_values = [], []
        for auxiliary in (False, True):
            seed_everything(42)
            model = SmallResNet18(100, auxiliary, seed=42)
            hashes.append(model.shared_state_hash())
            random_values.append(torch.rand(7))
        self.assertEqual(hashes[0], hashes[1])
        torch.testing.assert_close(*random_values)

    def test_small_image_feature_maps_and_identical_inference(self):
        """Retain valid cue grids and discard auxiliary computation at inference."""
        for size, classes, spatial in ((32, 100, 4), (64, 200, 8)):
            seed_everything(0)
            model = SmallResNet18(classes, auxiliary=True).eval()
            image = torch.randn(2, 3, size, size)
            with torch.no_grad():
                previous, final = model.feature_maps(image)
                self.assertEqual(final.shape, (2, 512, spatial, spatial))
                global_, _, votes = model.zipf_predictions(previous, final)
                torch.testing.assert_close(model(image), global_)
                self.assertEqual(votes.shape[1], 4 * (spatial**2 + (2 * spatial) ** 2))
                local, areas = model.readout.cue_logits(final)
                torch.testing.assert_close(
                    (local * areas[None, :, None]).sum(1), global_, rtol=1e-5, atol=1e-5
                )

    def test_all_training_methods_reach_backbone_and_auxiliary(self):
        """Every intervention must produce finite updates and correct auxiliary gradients."""
        for method in (
            "erm",
            "label_smoothing",
            "fixed_global_local",
            "zipfls_dense2",
            "cue",
            "pure_marginal",
            "image_mean_gate",
            "permuted_gate",
        ):
            seed_everything(0)
            model = SmallResNet18(100, auxiliary=method == "zipfls_dense2")
            config = recipe("cifar100", method=method, debug=True, device="cpu", precision="fp32")
            loss, _ = training_objective(
                model, torch.randn(2, 3, 32, 32), torch.tensor([0, 99]), config
            )
            loss.backward()
            self.assertTrue(math.isfinite(float(loss.detach())))
            self.assertGreater(float(next(model.parameters()).grad.abs().sum()), 0)
            if method == "zipfls_dense2":
                self.assertGreater(float(model.auxiliary.weight.grad.abs().sum()), 0)


class ClassificationConfigTests(unittest.TestCase):
    """Protect validated editable recipes and the explicit debug boundary."""

    def test_roundtrip_and_dataset_defaults(self):
        """Preserve complete settings and select the declared Tiny auxiliary weight."""
        for dataset in ("cifar100", "tiny_imagenet", "dominoes"):
            value = recipe(dataset)
            self.assertEqual(Config.from_dict(value.to_dict()), value)
        self.assertEqual(recipe("tiny_imagenet").zipf_aux_weight, 0.5)

    def test_invalid_or_silent_recipe_changes_rejected(self):
        """Accept explicit production changes while rejecting invalid values and hidden reductions."""
        customized = replace(
            Config(), epochs=5, milestones=(2, 4), batch_size=64, label_smoothing=0.2
        ).validate()
        self.assertEqual(customized.epochs, 5)
        self.assertEqual(customized.milestones, (2, 4))
        self.assertEqual(customized.batch_size, 64)
        self.assertEqual(customized.label_smoothing, 0.2)
        self.assertEqual(Config.from_dict(customized.to_dict()), customized)
        for updates in (
            {"epochs": 5},
            {"batch_size": 1},
            {"eval_batch_size": 0},
            {"seed": True},
            {"gate_tau": math.nan},
            {"learning_rate": -0.1},
            {"milestones": (150, 100)},
            {"workers": -1},
            {"device": "cpu"},
            {"debug_max_batches": 1},
        ):
            with self.assertRaises((ValueError, TypeError)):
                replace(Config(), **updates).validate()
        with self.assertRaises(TypeError):
            Config.from_dict({"unrecognized": 1})
