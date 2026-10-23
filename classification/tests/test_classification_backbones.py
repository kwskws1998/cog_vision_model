"""Independent reference and compatibility checks for selectable ResNet activation layouts."""

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

import torch
from clvision.config import recipe
from clvision.losses import training_objective
from clvision.models import SmallResNet18, ZipfPreActBlock
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from cue_routing.models import CueReadout


def pinned_reference():
    """Load the vendored Python-only reference without downloading weights or executing a CLI."""
    path = (
        Path(__file__).resolve().parents[1]
        / "references/zipfls/pytorch_zipfls__cifar_models__resnet.py.txt"
    )
    loader = importlib.machinery.SourceFileLoader("zipf_backbone_test_reference", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def preact_reference_name(name):
    """Map only the shared backbone and global head to the independently executed reference."""
    prefixes = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
        "readout.classifier.": "linear.",
    }
    for actual, expected in prefixes.items():
        if name.startswith(actual):
            return expected + name[len(actual) :]
    raise AssertionError(f"Unexpected shared parameter or buffer: {name}")


def legacy_postact(classes, auxiliary, seed):
    """Preserve the pre-selector constructor as an independent initialization/RNG oracle."""
    network = resnet18(weights=None, num_classes=classes)
    network.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(network.conv1.weight, mode="fan_out", nonlinearity="relu")
    network.maxpool = nn.Identity()
    model = nn.Module()
    model.backbone = nn.Sequential(
        network.conv1,
        network.bn1,
        network.relu,
        network.maxpool,
        network.layer1,
        network.layer2,
        network.layer3,
        network.layer4,
    )
    model.readout = CueReadout(512, classes, grid_size=2)
    model.readout.classifier = network.fc
    model.auxiliary = None
    if auxiliary:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed + 32017).get_state())
            model.auxiliary = nn.Linear(256, classes)
    return model


class BackboneTests(unittest.TestCase):
    """Exercise both architectures without benchmark downloads or CUDA allocation."""

    def setUp(self):
        """Isolate global RNG and bound CPU thread overhead for small synthetic batches."""
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(2)
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(846)

    def test_default_and_explicit_postact_preserve_legacy_state_rng_and_outputs(self):
        """Existing runs retain the same initialization, key layout, RNG draws, and predictions."""
        torch.manual_seed(1729)
        reference = legacy_postact(7, auxiliary=True, seed=19).eval()
        expected_rng = torch.get_rng_state()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected_logits = reference.readout(reference.backbone(images))
        for kwargs in ({}, {"activation": "postact"}):
            with self.subTest(kwargs=kwargs):
                torch.manual_seed(1729)
                actual = SmallResNet18(7, auxiliary=True, seed=19, **kwargs).eval()
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(set(actual.state_dict()), set(reference.state_dict()))
                for name, value in actual.state_dict().items():
                    self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
                with torch.no_grad():
                    torch.testing.assert_close(actual(images), expected_logits, rtol=0, atol=0)

    def test_preact_matches_official_initial_state_logits_gradients_and_bn_updates(self):
        """Compare the full 32px train graph with the pinned reference, not a restated formula."""
        module = pinned_reference()
        torch.manual_seed(1729)
        actual = SmallResNet18(7, activation="preact").train()
        torch.manual_seed(1729)
        expected = module.CIFAR_ResNet18(num_classes=7).train()
        for name, value in actual.state_dict().items():
            with self.subTest(initial=name):
                self.assertTrue(
                    torch.equal(value, expected.state_dict()[preact_reference_name(name)])
                )
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        reference_images = images.detach().clone().requires_grad_()
        targets = torch.tensor([0, 6])
        actual_logits = actual(images)
        expected_logits = expected(reference_images)[0][0]
        torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-6)
        F.cross_entropy(actual_logits, targets).backward()
        F.cross_entropy(expected_logits, targets).backward()
        torch.testing.assert_close(images.grad, reference_images.grad, rtol=2e-4, atol=2e-6)
        reference_parameters = dict(expected.named_parameters())
        self.assertEqual(
            {preact_reference_name(name) for name, _ in actual.named_parameters()},
            {
                name
                for name in reference_parameters
                if not name.startswith(("linear2.", "linear3."))
            },
        )
        for name, parameter in actual.named_parameters():
            with self.subTest(gradient=name):
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(
                    parameter.grad,
                    reference_parameters[preact_reference_name(name)].grad,
                    rtol=2e-4,
                    atol=2e-6,
                )
        for name, value in actual.named_buffers():
            with self.subTest(buffer=name):
                torch.testing.assert_close(
                    value, expected.state_dict()[preact_reference_name(name)], rtol=0, atol=0
                )

    def test_preact_identity_shortcut_uses_activated_input(self):
        """Guard the official Zipf-specific shortcut against silently substituting textbook v2."""
        block = ZipfPreActBlock(2, 2).eval()
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            values = torch.tensor([-2.0, 3.0]).reshape(1, 2, 1, 1).expand(1, 2, 4, 4)
            expected = F.relu(block.bn1(values))
            torch.testing.assert_close(block(values), expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(expected, values))

    def test_both_backbones_support_resolutions_rectangles_and_zipf_heads(self):
        """Preserve spatial heads, signed PreAct features, and GAP equivalence beyond 32px."""
        for activation in ("postact", "preact"):
            model = SmallResNet18(7, auxiliary=True, activation=activation).eval()
            for height, width in ((32, 32), (64, 64), (40, 56)):
                with self.subTest(activation=activation, resolution=(height, width)):
                    with torch.no_grad():
                        penultimate, final = model.feature_maps(torch.randn(1, 3, height, width))
                        self.assertEqual(penultimate.shape, (1, 256, height // 4, width // 4))
                        self.assertEqual(final.shape, (1, 512, height // 8, width // 8))
                        if activation == "preact":
                            self.assertTrue(bool((final < 0).any()))
                        local, areas = model.readout.cue_logits(final)
                        global_logits, auxiliary_logits, votes = model.zipf_predictions(
                            penultimate, final
                        )
                        torch.testing.assert_close(
                            (local * areas[None, :, None]).sum(1),
                            model.readout(final),
                            rtol=2e-5,
                            atol=2e-6,
                        )
                        torch.testing.assert_close(global_logits, model.readout(final))
                        self.assertEqual(auxiliary_logits.shape, (1, 7))
                        expected_votes = 4 * (
                            final.shape[2] * final.shape[3]
                            + penultimate.shape[2] * penultimate.shape[3]
                        )
                        self.assertEqual(votes.shape, (1, expected_votes))

    def test_signed_unequal_area_cues_preserve_global_head_gradients(self):
        """Negative features and odd map sizes retain the exact area-weighted readout identity."""
        readout = CueReadout(6, 7).double()
        features = torch.randn(2, 6, 5, 7, dtype=torch.float64, requires_grad=True)
        self.assertTrue(bool((features < 0).any()))
        local_logits, areas = readout.cue_logits(features)
        joint = (local_logits * areas[None, :, None]).sum(1)
        direct = readout(features)
        torch.testing.assert_close(joint, direct, rtol=1e-12, atol=1e-12)
        parameters = (features, readout.classifier.weight, readout.classifier.bias)
        joint_gradients = torch.autograd.grad(joint.square().sum(), parameters, retain_graph=True)
        direct_gradients = torch.autograd.grad(direct.square().sum(), parameters)
        for actual, expected in zip(joint_gradients, direct_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_auxiliary_heads_do_not_change_shared_initialization_or_rng(self):
        """ERM/CUE and Zipf receive identical shared weights and subsequent data-order RNG."""
        for activation in ("postact", "preact"):
            results = []
            for auxiliary in (False, True):
                torch.manual_seed(1729)
                model = SmallResNet18(7, auxiliary=auxiliary, seed=19, activation=activation)
                results.append((model.shared_state_hash(), torch.get_rng_state()))
            with self.subTest(activation=activation):
                self.assertEqual(results[0][0], results[1][0])
                self.assertTrue(torch.equal(results[0][1], results[1][1]))

    def test_preact_training_methods_reach_shared_and_auxiliary_parameters(self):
        """Every existing intervention must backpropagate finite updates through the new layout."""
        images = torch.randn(2, 3, 32, 32)
        targets = torch.tensor([0, 6])
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
            with self.subTest(method=method):
                model = SmallResNet18(
                    7, auxiliary=method == "zipfls_dense2", activation="preact"
                ).train()
                config = recipe(
                    "cifar100", method=method, device="cpu", precision="fp32", debug=True
                )
                loss, _ = training_objective(model, images, targets, config)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                self.assertGreater(float(model.backbone[0].weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.readout.classifier.weight.grad.abs().sum()), 0)
                if method == "zipfls_dense2":
                    self.assertGreater(float(model.auxiliary.weight.grad.abs().sum()), 0)

    def test_unknown_activation_fails_before_consuming_rng(self):
        """Reject unsupported layouts instead of silently falling back to a different backbone."""
        before = torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "activation"):
            SmallResNet18(7, activation="invalid")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
