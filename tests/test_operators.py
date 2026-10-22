"""Independent numerical tests of the proposed gradient and the pinned RSC masks."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cue_routing.models import CueReadout, ResNet18Classifier
from cue_routing.routing import route_gates, routed_loss
from cue_routing.rsc import rsc_logits


class RoutingTests(unittest.TestCase):
    """Test algebraic identities rather than merely replaying the implementation."""

    def setUp(self):
        torch.manual_seed(37)
        self.logits = torch.randn(5, 4, 7, dtype=torch.double, requires_grad=True)
        self.weights = torch.tensor([9, 12, 12, 16], dtype=torch.double) / 49
        self.targets = torch.tensor([0, 2, 4, 5, 6])

    def test_exact_per_cue_gradient(self):
        gates = torch.rand(5, 4, dtype=torch.double)
        loss, _, _ = routed_loss(self.logits, self.targets, self.weights, gates_override=gates)
        actual = torch.autograd.grad(loss, self.logits)[0]
        joint = (self.logits.detach() * self.weights[None, :, None]).sum(1).softmax(-1)
        local = self.logits.detach().softmax(-1)
        label = F.one_hot(self.targets, 7).double()[:, None]
        expected = (
            self.weights[None, :, None]
            * (
                gates[:, :, None] * (joint[:, None] - label)
                + (1 - gates[:, :, None]) * (local - label)
            )
            / 5
        )
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)

    def test_all_joint_equals_erm_through_shared_parameters(self):
        readout = CueReadout(3, 7).double()
        features = torch.randn(5, 3, 7, 7, dtype=torch.double, requires_grad=True)
        local, weights = readout.cue_logits(features)
        routed, _, _ = routed_loss(
            local, self.targets, weights, gates_override=torch.ones(5, 4, dtype=torch.double)
        )
        parameters = [features, readout.classifier.weight, readout.classifier.bias]
        first = torch.autograd.grad(routed, parameters, retain_graph=True)
        ordinary = F.cross_entropy(readout(features), self.targets)
        second = torch.autograd.grad(ordinary, parameters)
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)

    def test_marginal_minus_joint_is_weighted_kl(self):
        joint_logits = (self.logits * self.weights[None, :, None]).sum(1)
        log_local, joint = self.logits.log_softmax(-1), joint_logits.softmax(-1)
        marginal = -(
            log_local.gather(2, self.targets[:, None, None].expand(5, 4, 1)).squeeze(-1)
            * self.weights
        ).sum(1)
        joint_ce = F.cross_entropy(joint_logits, self.targets, reduction="none")
        kl = (joint[:, None] * (joint.log()[:, None] - log_local)).sum(-1)
        torch.testing.assert_close(
            marginal - joint_ce, (kl * self.weights).sum(1), atol=1e-14, rtol=1e-12
        )

    def test_logit_gauge_invariance(self):
        shift = torch.randn(5, 4, 1, dtype=torch.double) * 50
        original = route_gates(self.logits, self.weights)
        shifted = route_gates(self.logits + shift, self.weights)
        torch.testing.assert_close(original[0], shifted[0], atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(original[1], shifted[1], atol=1e-13, rtol=1e-12)
        first = routed_loss(self.logits, self.targets, self.weights)[0]
        second = routed_loss(self.logits + shift, self.targets, self.weights)[0]
        torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-12)

    def test_single_cue_is_erm(self):
        local = self.logits[:, :1].detach().requires_grad_(True)
        loss, _, _ = routed_loss(local, self.targets, torch.ones(1, dtype=torch.double))
        expected = F.cross_entropy(local[:, 0], self.targets)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(loss, local, retain_graph=True)[0],
            torch.autograd.grad(expected, local)[0],
        )

    def test_large_opposing_logits_preserve_joint_surrogate(self):
        local = torch.tensor([[[1e8, 1e8, 0.0], [-1e8 + 8, -1e8, 0.0]]], requires_grad=True)
        weights, labels = torch.tensor([0.5, 0.5]), torch.tensor([2])
        loss, joint, _ = routed_loss(local, labels, weights, gates_override=torch.ones(1, 2))
        actual = torch.autograd.grad(loss, local)[0]
        expected = weights[None, :, None] * (
            joint.softmax(-1)[:, None] - F.one_hot(labels, 3)[:, None]
        )
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_gates_detached_bounded_and_modes(self):
        for mode in ("cue", "image", "fixed", "shuffled"):
            gates, disagreement, _ = route_gates(self.logits, self.weights, mode=mode)
            self.assertFalse(gates.requires_grad)
            self.assertTrue(bool(((gates >= 0.5) & (gates <= 1)).all()))
            self.assertTrue(bool(((disagreement >= 0) & (disagreement <= 1)).all()))
            if mode == "image":
                torch.testing.assert_close(gates, gates[:, :1].expand_as(gates))
            if mode == "fixed":
                torch.testing.assert_close(gates, torch.full_like(gates, 0.75))

    def test_per_cue_gate_is_not_image_average(self):
        gates = torch.tensor([[0.1, 0.3, 0.7, 0.9]], dtype=torch.double).expand(5, -1)
        local_loss = routed_loss(self.logits, self.targets, self.weights, gates_override=gates)[0]
        averaged = (gates * self.weights).sum(1, keepdim=True).expand_as(gates)
        average_loss = routed_loss(
            self.logits, self.targets, self.weights, gates_override=averaged
        )[0]
        first = torch.autograd.grad(local_loss, self.logits, retain_graph=True)[0]
        second = torch.autograd.grad(average_loss, self.logits)[0]
        self.assertGreater(float((first - second).abs().max()), 1e-3)

    def test_invalid_weights_labels_and_nonfinite(self):
        for weights in (self.weights * 2, self.weights * float("nan"), -self.weights):
            with self.assertRaises(ValueError):
                routed_loss(self.logits, self.targets, weights)
        with self.assertRaises(ValueError):
            routed_loss(self.logits, self.targets + 10, self.weights)


class ReadoutTests(unittest.TestCase):
    """Check CNN and ViT compatibility with no added inference architecture."""

    def test_odd_map_area_preservation_and_tokens(self):
        torch.manual_seed(1)
        head = CueReadout(3, 7).double()
        maps = torch.randn(2, 3, 7, 7, dtype=torch.double)
        local, weights = head.cue_logits(maps)
        expected_weights = torch.tensor([9, 12, 12, 16], dtype=torch.double) / 49
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(
            (local * weights[None, :, None]).sum(1), head(maps), atol=1e-14, rtol=1e-12
        )
        tokens = maps.flatten(2).transpose(1, 2)
        torch.testing.assert_close(head(tokens, token_grid=(7, 7)), head(maps))
        with self.assertRaises(ValueError):
            head(torch.cat([tokens[:, :1], tokens], dim=1), token_grid=(7, 7))

    def test_frozen_batchnorm_and_inference(self):
        model = ResNet18Classifier(pretrained=False).train()
        bn = next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))
        self.assertFalse(bn.training)
        self.assertTrue(bn.weight.requires_grad)
        before = bn.running_mean.clone()
        with torch.no_grad():
            images = torch.randn(2, 3, 64, 64)
            output = model(images)
            maps = model.forward_features(images)
            expected = model.classifier(maps.mean((2, 3)))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(before, bn.running_mean)


class RSCTests(unittest.TestCase):
    """Compare masks against an independent NumPy percentile reference, including ties."""

    def reference(self, features, labels, head, feature_drop, batch_drop):
        """Reproduce the pinned algorithm using closed-form linear-head derivatives."""
        values = features.detach().numpy()
        weights, bias = head.weight.detach().numpy(), head.bias.detach().numpy()
        logits = values @ weights.T + bias
        gradients = weights[labels.numpy()]
        mask_f = gradients < np.percentile(
            gradients, 100 * (1 - feature_drop), axis=1, keepdims=True
        ).astype(gradients.dtype)
        muted = (values * mask_f) @ weights.T + bias

        def softmax(value):
            """Compute an independent stable row-wise NumPy softmax."""
            exponential = np.exp(value - value.max(1, keepdims=True))
            return exponential / exponential.sum(1, keepdims=True)

        changes = (
            softmax(logits)[np.arange(len(labels)), labels.numpy()]
            - softmax(muted)[np.arange(len(labels)), labels.numpy()]
        )
        mask_b = changes < np.percentile(changes, 100 * (1 - batch_drop)).astype(changes.dtype)
        return np.logical_or(mask_f, mask_b[:, None])

    def test_mask_parity_and_ties(self):
        torch.manual_seed(82)
        for dtype, tied in (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float64, False),
            (torch.float64, True),
        ):
            head = nn.Linear(12, 7).to(dtype)
            features = torch.randn(9, 12, dtype=dtype, requires_grad=True)
            labels = torch.arange(9) % 7
            if tied:
                with torch.no_grad():
                    head.weight.fill_(0.25)
                    head.bias.zero_()
                    features.zero_()
            challenged, _, mask, _ = rsc_logits(features, labels, head)
            expected = self.reference(features, labels, head, 1 / 3, 1 / 3)
            np.testing.assert_array_equal(mask.numpy(), expected)
            loss = F.cross_entropy(challenged, labels)
            loss.backward()
            self.assertIsNotNone(head.weight.grad)
            self.assertIsNotNone(features.grad)
            self.assertTrue(bool(torch.isfinite(features.grad).all()))

    def test_backbone_gradient_survives_mask_gradient_query(self):
        encoder, head = nn.Linear(5, 12), nn.Linear(12, 7)
        features = encoder(torch.randn(9, 5))
        challenged, _, _, _ = rsc_logits(features, torch.arange(9) % 7, head)
        F.cross_entropy(challenged, torch.arange(9) % 7).backward()
        self.assertGreater(float(encoder.weight.grad.abs().sum()), 0)
