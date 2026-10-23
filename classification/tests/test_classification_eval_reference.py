"""Independent vanilla ResNet checks for ERM gradients and validation semantics."""

import unittest

import torch
from clvision.config import recipe
from clvision.engine import evaluate_loader
from clvision.losses import training_objective
from clvision.models import SmallResNet18
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def reference_name(name):
    """Map the explicit shared architecture to torchvision's independently executed graph."""
    prefixes = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
        "readout.classifier.": "fc.",
    }
    for actual, expected in prefixes.items():
        if name.startswith(actual):
            return expected + name[len(actual) :]
    raise AssertionError(f"Unexpected ERM parameter or buffer: {name}")


def vanilla_reference(model):
    """Copy weights into a plain torchvision ResNet with only the prescribed small-image stem."""
    reference = resnet18(weights=None, num_classes=100)
    reference.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    reference.maxpool = nn.Identity()
    reference.load_state_dict(
        {reference_name(name): value for name, value in model.state_dict().items()}, strict=True
    )
    return reference


class EvaluationReferenceTests(unittest.TestCase):
    """Verify the complete ERM and evaluation paths without requiring benchmark data or CUDA."""

    def setUp(self):
        """Isolate RNG and thread settings while creating small deterministic synthetic batches."""
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(773)
        self.images = torch.randn(5, 3, 32, 32)
        self.targets = torch.tensor([0, 17, 99, 31, 8], dtype=torch.long)
        self.config = recipe(
            "cifar100",
            method="erm",
            device="cpu",
            precision="fp32",
            eval_precision="fp32",
            debug=True,
            workers=0,
        )

    def test_erm_logits_loss_and_all_gradients_match_vanilla_resnet(self):
        """ERM must equal ordinary global-average ResNet CE through every shared parameter."""
        model = SmallResNet18(100).train()
        reference = vanilla_reference(model).train()
        actual_logits = model(self.images)
        expected_logits = reference(self.images)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-6, atol=1e-7)
        loss, diagnostics = training_objective(model, self.images, self.targets, self.config)
        expected_loss = F.cross_entropy(reference(self.images), self.targets)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            diagnostics["train_accuracy"],
            expected_logits.detach().argmax(1).eq(self.targets).float().mean(),
        )
        loss.backward()
        expected_loss.backward()
        expected_parameters = dict(reference.named_parameters())
        self.assertEqual(
            {reference_name(name) for name, _ in model.named_parameters()},
            set(expected_parameters),
        )
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                expected = expected_parameters[reference_name(name)]
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(expected.grad)
                torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-6, atol=1e-7)

    def test_evaluation_preserves_bn_and_matches_batch_independent_reference(self):
        """Validation must use frozen BN, retain all examples, and compute ordinary accuracy."""
        model = SmallResNet18(100).train()
        with torch.no_grad():
            model(self.images)
        reference = vanilla_reference(model).eval()
        with torch.no_grad():
            expected_logits = reference(self.images)
        targets = expected_logits.argmax(1)
        targets[1::2] = (targets[1::2] + 1) % 100
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        metrics, logits, labels = evaluate_loader(model, [(self.images, targets)], self.config)
        split_metrics, split_logits, split_labels = evaluate_loader(
            model,
            [(self.images[:2], targets[:2]), (self.images[2:], targets[2:])],
            self.config,
        )
        self.assertTrue(all(not module.training for module in model.modules()))
        for name, value in model.named_buffers():
            with self.subTest(buffer=name):
                self.assertTrue(torch.equal(value, buffers[name]))
        torch.testing.assert_close(logits, expected_logits, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(split_logits, expected_logits, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(labels, targets))
        self.assertTrue(torch.equal(split_labels, targets))
        self.assertFalse(logits.requires_grad)
        expected_accuracy = float(expected_logits.argmax(1).eq(targets).double().mean())
        expected_nll = float(F.cross_entropy(expected_logits.double(), targets))
        self.assertEqual(expected_accuracy, 0.6)
        for result in (metrics, split_metrics):
            self.assertEqual(result["n"], len(targets))
            self.assertEqual(result["accuracy"], expected_accuracy)
            self.assertAlmostEqual(result["nll"], expected_nll, places=6)
