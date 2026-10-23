"""Regression checks for evaluation overflow, precise failures, and preflight worker cleanup."""

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from clvision.arguments import add_study_arguments, load_study_arguments, study_argument_tokens
from clvision.config import Config, recipe
from clvision.engine import evaluate_loader, optimize_batch
from clvision.profile import memory_signature, profile_dataset, profile_iterator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cue_routing.io import read_json
from cue_routing.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[2]


class ClassificationNumericsTests(unittest.TestCase):
    """Exercise real overflow and invalid-state paths without claiming a CUDA reproduction."""

    def config(self, **updates):
        """Use a valid CPU debug recipe for hardware-independent numerical checks."""
        return recipe("cifar100", device="cpu", precision="fp32", debug=True, workers=0, **updates)

    def test_fp32_evaluation_survives_fp16_overflow_and_ambient_autocast(self):
        """The actual half-precision matmul overflows while explicitly FP32 evaluation remains finite."""
        model = nn.Linear(1, 100, bias=False)
        with torch.no_grad():
            model.weight.fill_(80000)
        images, targets = torch.ones(2, 1), torch.tensor([0, 99])
        before = model.weight.detach().clone()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
            self.assertFalse(bool(torch.isfinite(model(images)).all()))
            metrics, logits, labels = evaluate_loader(model, [(images, targets)], self.config())
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(torch.equal(logits, torch.full((2, 100), 80000.0)))
        self.assertTrue(torch.equal(labels, targets))
        self.assertTrue(torch.equal(before, model.weight))
        self.assertEqual(metrics["n"], 2)

    def test_nonfinite_predictions_identify_state_and_recipe(self):
        """FP32 failure remains fatal and names the offending model state without replacing logits."""
        model = nn.Linear(1, 100, bias=False)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError) as caught:
            evaluate_loader(model, [(torch.ones(2, 1), torch.tensor([0, 1]))], self.config())
        message = str(caught.exception)
        for detail in (
            "dataset=cifar100",
            "method=cue",
            "eval_precision=fp32",
            "batch=1",
            "nan=2",
            "nonfinite_inputs=0",
            "nonfinite_model_tensors=['weight']",
        ):
            self.assertIn(detail, message)

    def test_invalid_labels_are_rejected_before_forward(self):
        """Distinguish bad class IDs and non-integer labels from a prediction overflow."""
        for targets in (torch.tensor([0, 100]), torch.tensor([-1, 2]), torch.tensor([0.0, 1.0])):
            model = Mock()
            with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, "label"):
                evaluate_loader(model, [(torch.ones(2, 1), targets)], self.config())
            model.assert_not_called()

    def test_nonfinite_running_buffer_stops_before_optimizer_step(self):
        """A finite loss cannot authorize an update when a running statistic has become NaN."""
        model = nn.Linear(1, 1)
        model.register_buffer("running_var", torch.ones(1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        before = model.weight.detach().clone()

        def objective(model, images, targets, config):
            """Simulate a damaged running statistic alongside a differentiable finite loss."""
            model.running_var.fill_(float("nan"))
            return model(images).square().mean(), {}

        with patch("clvision.engine.training_objective", side_effect=objective):
            with self.assertRaisesRegex(FloatingPointError, "buffers.*running_var"):
                optimize_batch(
                    model,
                    optimizer,
                    torch.amp.GradScaler("cpu", enabled=False),
                    torch.ones(2, 1),
                    torch.zeros(2, dtype=torch.long),
                    self.config(),
                )
        self.assertTrue(torch.equal(before, model.weight))
        self.assertFalse(optimizer.state)

    def test_metrics_distinguish_nonfinite_logits_and_invalid_labels(self):
        """The shared metrics no longer collapse unrelated failures into one generic error."""
        with self.assertRaisesRegex(ValueError, "Nonfinite logits: nan=1, inf=1"):
            classification_metrics(torch.tensor([[float("nan"), float("inf")]]), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, r"Labels outside \[0, 1\]: min=2, max=2"):
            classification_metrics(torch.zeros(1, 2), torch.tensor([2]))

    def test_evaluation_precision_is_validated_forwarded_and_bound(self):
        """Training/evaluation choices resolve independently and invalidate stale profile evidence."""
        parser = argparse.ArgumentParser()
        add_study_arguments(parser, ROOT)
        arguments = parser.parse_args(["--precision", "fp16", "--eval-precision", "bf16"])
        config = load_study_arguments(arguments).recipe("cifar100")
        self.assertEqual((config.precision, config.eval_precision), ("fp16", "bf16"))
        tokens = study_argument_tokens(arguments)
        self.assertEqual(tokens[tokens.index("--eval-precision") + 1], "bf16")
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        self.assertNotEqual(
            memory_signature(config), memory_signature(replace(config, eval_precision="fp32"))
        )
        with self.assertRaisesRegex(ValueError, "evaluation precision"):
            replace(config, eval_precision="invalid").validate()
        with self.assertRaisesRegex(ValueError, "FP32 debug"):
            replace(self.config(), eval_precision="fp16").validate()

    def test_failed_preflight_preserves_stage_and_successful_methods(self):
        """A failed validation retains the prior method and records the exact accepted-update count."""

        def measured(manifest, config, steps, activity):
            """Simulate CUDA measurement results only to test artifact handling and error propagation."""
            activity.update(stage="validation", accepted_optimizer_steps=10)
            if config.method == "label_smoothing":
                raise FloatingPointError("Nonfinite evaluation logits: nan=3")
            return {
                "training_images_per_second": 100,
                "evaluation_images_per_second": 200,
                "vram": {"peak_reserved_gib": 1},
            }

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "profile.json"
            with (
                patch("clvision.profile.environment", return_value={"test_only": True}),
                patch("clvision.profile.torch.cuda.is_available", return_value=True),
                patch("clvision.profile.profile_method", side_effect=measured),
                self.assertRaisesRegex(
                    RuntimeError, "method=label_smoothing stage=validation accepted_updates=10/10"
                ),
            ):
                profile_dataset(ROOT, {"dataset_id": "fixture"}, recipe("cifar100"), destination)
            result = read_json(destination)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["measured"])
        self.assertFalse(result["passes_budget"])
        self.assertEqual(set(result["methods"]), {"erm"})
        self.assertEqual(result["active"]["config"]["eval_precision"], "fp32")
        self.assertEqual(result["error"]["type"], "FloatingPointError")

    def test_preflight_partial_loader_workers_close_on_failure(self):
        """A real multiprocessing loader is shut down before the original exception escapes."""
        loader = DataLoader(TensorDataset(torch.ones(16, 1)), batch_size=2, num_workers=1)
        with self.assertRaisesRegex(ValueError, "evaluation failed"):
            with profile_iterator(loader) as iterator:
                next(iterator)
                workers = list(iterator._workers)
                raise ValueError("evaluation failed")
        self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_preflight_single_process_iterator_needs_no_shutdown(self):
        """The same preflight lifecycle supports workers=0 and ordinary iterables."""
        with profile_iterator([1, 2]) as iterator:
            self.assertEqual(next(iterator), 1)


if __name__ == "__main__":
    unittest.main()
