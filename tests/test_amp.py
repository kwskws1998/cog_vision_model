"""Actual GradScaler regressions for overflow recovery, replay, and fatal failures."""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_protocol import TinyClassifier

from cue_routing.cli import real_data_smoke
from cue_routing.config import METHODS, ExperimentConfig
from cue_routing.data import evaluation_loader, load_manifest, make_fixture
from cue_routing.engine import optimize_batch
from cue_routing.io import capture_rng, restore_rng, seed_everything


class HalfBoundaryProbe(torch.nn.Module):
    """Force scaled backward overflow at a real FP16 boundary on a CPU tensor."""

    def __init__(self):
        """Keep one trainable weight and one mutable forward buffer."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))

    def forward(self, images):
        """Consume three RNGs and mutate a buffer before a genuine half cast."""
        self.forward_count.add_(1)
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        return (self.weight * images + draw).half().float(), draw


class AMPRecoveryTests(unittest.TestCase):
    """Exercise the real CPU GradScaler rather than mocking its overflow state."""

    def test_overflow_replays_same_batch_rng_buffers_and_one_adam_update(self):
        for method in METHODS:
            with self.subTest(method=method):
                seed_everything(37)
                config = ExperimentConfig(method=method, device="cpu", precision="fp32")
                model = HalfBoundaryProbe()
                reference = copy.deepcopy(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
                scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
                images, labels = torch.ones(3), torch.zeros(3, dtype=torch.long)
                initial_rng = capture_rng()
                attempts = []

                def objective(current, batch, targets, recipe):
                    """Retain each real replay's inputs and random draw for comparison."""
                    predictions, draw = current(batch)
                    attempts.append((batch.clone(), targets.clone(), draw))
                    return predictions.square().mean(), {"random_draw": draw}

                with (
                    patch("cue_routing.engine.training_objective", side_effect=objective),
                    patch.object(optimizer, "step", wraps=optimizer.step) as steps,
                ):
                    result = optimize_batch(model, optimizer, scaler, images, labels, config)
                self.assertGreater(result["amp_overflow_retries"], 0)
                self.assertEqual(len(attempts), result["amp_overflow_retries"] + 1)
                self.assertEqual(steps.call_count, 1)
                self.assertEqual(float(optimizer.state[model.weight]["step"]), 1.0)
                self.assertEqual(int(model.forward_count), 1)
                recovered_rng = capture_rng()
                for batch, targets, draw in attempts:
                    self.assertTrue(torch.equal(batch, images))
                    self.assertTrue(torch.equal(targets, labels))
                    self.assertEqual(draw, attempts[0][2])
                restore_rng(initial_rng)
                reference_scaler = torch.amp.GradScaler("cpu", init_scale=result["amp_scale"])
                with patch("cue_routing.engine.training_objective", side_effect=objective):
                    direct = optimize_batch(
                        reference, reference_optimizer, reference_scaler, images, labels, config
                    )
                self.assertEqual(direct["amp_overflow_retries"], 0)
                self.assertEqual(result["loss"], direct["loss"])
                self.assertTrue(torch.equal(model.weight, reference.weight))
                for name, value in optimizer.state[model.weight].items():
                    self.assertTrue(
                        torch.equal(value, reference_optimizer.state[reference.weight][name])
                    )
                direct_rng = capture_rng()
                self.assertEqual(recovered_rng["python"], direct_rng["python"])
                self.assertEqual(recovered_rng["numpy"], direct_rng["numpy"])
                self.assertTrue(torch.equal(recovered_rng["torch"], direct_rng["torch"]))

    def test_persistent_nonfinite_gradients_are_bounded_and_never_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        before = model.weight.detach().clone()
        config = ExperimentConfig(device="cpu", precision="fp32")

        def objective(current, images, labels, recipe):
            """Produce a finite forward with an intrinsically infinite derivative."""
            return (current.weight - current.weight.detach()).sqrt().sum(), {}

        with (
            patch("cue_routing.engine.training_objective", side_effect=objective) as calls,
            patch.object(optimizer, "step", wraps=optimizer.step) as steps,
        ):
            with self.assertRaisesRegex(FloatingPointError, "after 17 AMP attempts"):
                optimize_batch(model, optimizer, scaler, torch.ones(1), torch.zeros(1), config)
        self.assertEqual(calls.call_count, 17)
        self.assertEqual(steps.call_count, 0)
        self.assertFalse(optimizer.state)
        self.assertTrue(torch.equal(model.weight, before))

    def test_disabled_scaler_nonfinite_gradients_remain_fatal(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        config = ExperimentConfig(device="cpu", precision="fp32")
        loss = (model.weight - model.weight.detach()).sqrt().sum()
        with patch("cue_routing.engine.training_objective", return_value=(loss, {})):
            with self.assertRaisesRegex(FloatingPointError, "without enabled FP16"):
                optimize_batch(model, optimizer, scaler, torch.ones(1), torch.zeros(1), config)
        self.assertFalse(optimizer.state)

    def test_nonfinite_forward_loss_never_updates_or_retries(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        config = ExperimentConfig(device="cpu", precision="fp32")
        loss = model.weight.sum() * torch.tensor(float("inf"))
        with patch("cue_routing.engine.training_objective", return_value=(loss, {})) as calls:
            with self.assertRaisesRegex(FloatingPointError, "forward pass"):
                optimize_batch(model, optimizer, scaler, torch.ones(1), torch.zeros(1), config)
        self.assertEqual(calls.call_count, 1)
        self.assertFalse(optimizer.state)

    def test_scaler_checkpoint_restores_recovery_state_for_the_next_update(self):
        seed_everything(11)
        config = ExperimentConfig(device="cpu", precision="fp32")
        model = HalfBoundaryProbe()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu")

        def objective(current, images, labels, recipe):
            """Exercise stochastic FP16 casts across a saved scaler checkpoint."""
            predictions, _ = current(images)
            return predictions.square().mean(), {}

        images, labels = torch.ones(3), torch.zeros(3, dtype=torch.long)
        with patch("cue_routing.engine.training_objective", side_effect=objective):
            optimize_batch(model, optimizer, scaler, images, labels, config)
            checkpoint = copy.deepcopy(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": capture_rng(),
                }
            )
            expected = optimize_batch(model, optimizer, scaler, images, labels, config)
            resumed = HalfBoundaryProbe()
            resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
            resumed_scaler = torch.amp.GradScaler("cpu")
            resumed.load_state_dict(checkpoint["model"])
            resumed_optimizer.load_state_dict(checkpoint["optimizer"])
            resumed_scaler.load_state_dict(checkpoint["scaler"])
            restore_rng(checkpoint["rng"])
            actual = optimize_batch(
                resumed, resumed_optimizer, resumed_scaler, images, labels, config
            )
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(model.weight, resumed.weight))
        self.assertTrue(torch.equal(model.forward_count, resumed.forward_count))
        self.assertEqual(scaler.state_dict(), resumed_scaler.state_dict())


class SourcePreflightTests(unittest.TestCase):
    """Check real-data preflight scheduling and source-only access using a labeled fixture."""

    def test_preflight_completes_matched_updates_without_target_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_fixture(root / "images", None)
            manifest = load_manifest(path, None)
            manifest["kind"] = "pacs"
            config = ExperimentConfig(
                device="cpu", precision="fp32", workers=0, batch_per_domain=1, eval_batch_size=3
            )
            with (
                patch("cue_routing.cli.load_manifest", return_value=manifest),
                patch("cue_routing.cli.make_model", side_effect=lambda *args: TinyClassifier()),
                patch("cue_routing.cli.source_hash", return_value="artificial-test-source"),
                patch("cue_routing.cli.evaluation_loader", wraps=evaluation_loader) as loaders,
            ):
                result = real_data_smoke(path, root, config, steps=2)
            self.assertEqual(loaders.call_count, 9)
            self.assertTrue(all(call.args[1] != config.target for call in loaders.call_args_list))
            self.assertFalse(result["target_loader_accessed"])
            self.assertFalse(result["benchmark_evidence"])
            self.assertTrue(result["matched_initial_weights"])
            self.assertTrue(result["matched_training_images_and_augmentations"])
            for method in METHODS:
                self.assertEqual(result["methods"][method]["optimizer_steps"], 2)
                self.assertEqual(len(result["methods"][method]["step_diagnostics"]), 2)

    def test_preflight_rejects_invalid_step_counts_before_data_access(self):
        with patch("cue_routing.cli.load_manifest") as data:
            for steps in (0, -1, True, 1.5):
                with (
                    self.subTest(steps=steps),
                    self.assertRaisesRegex(ValueError, "positive integer"),
                ):
                    real_data_smoke("unused", ".", steps=steps)
            data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
