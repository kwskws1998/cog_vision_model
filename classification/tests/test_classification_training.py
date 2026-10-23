"""Checkpoint replay, finite-update recovery, final access guards, and evaluation semantics."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from clvision.config import MAIN_METHODS, recipe
from clvision.data import make_fixture, make_loader, save_array
from clvision.engine import (
    dominoes_metrics,
    evaluate_corruptions,
    evaluate_run,
    optimize_batch,
    train,
    verify_run,
)
from torch import nn

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]


def debug_config(method="cue", dataset="cifar100"):
    """Use full ResNet operators on a deliberately tiny, explicitly marked verification run."""
    return recipe(
        dataset,
        method=method,
        device="cpu",
        precision="fp32",
        debug=True,
        workers=0,
        epochs=2,
        milestones=(1,),
        batch_size=2,
        eval_batch_size=4,
        debug_max_batches=1,
    )


def equal_states(left, right):
    """Require exact model and nested optimizer tensor identity across CPU resumes."""
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal_states(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_states(a, b) for a, b in zip(left, right))
    return left == right


class ClassificationTrainingTests(unittest.TestCase):
    """Check exact accepted-update behavior rather than inferring it from training logs."""

    def setUp(self):
        """Keep verification checkpoints in an isolated temporary directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.manifest = make_fixture(self.folder / "data")

    def tearDown(self):
        """Delete only files created by this test."""
        self.temporary.cleanup()

    def test_bitwise_resume_all_main_methods(self):
        """Model, momentum buffers, AMP state, and accepted counts must replay exactly."""
        for method in MAIN_METHODS:
            config = debug_config(method)
            full, resumed = self.folder / f"full_{method}", self.folder / f"resumed_{method}"
            train(config, self.manifest, full, ROOT, phase="debug")
            train(config, self.manifest, resumed, ROOT, phase="debug", stop_after=1)
            train(config, self.manifest, resumed, ROOT, phase="debug", resume=True)
            first = torch.load(full / "last.pt", weights_only=True)
            second = torch.load(resumed / "last.pt", weights_only=True)
            for key in ("model", "optimizer", "scaler", "rng", "steps", "epoch"):
                self.assertTrue(equal_states(first[key], second[key]), (method, key))
            self.assertAlmostEqual(first["history"][0]["learning_rate"], 0.1)
            self.assertAlmostEqual(first["history"][1]["learning_rate"], 0.01)
            self.assertEqual(first["steps"], 2)
            self.assertFalse(read_json(full / "summary.json")["benchmark_evidence"])

    def test_training_never_constructs_test_partition(self):
        """Validation-only training cannot instantiate a final-test loader."""
        partitions = []

        def traced(manifest, config, split, *args, **kwargs):
            partitions.append(split)
            return make_loader(manifest, config, split, *args, **kwargs)

        with patch("clvision.engine.make_loader", side_effect=traced):
            train(debug_config(), self.manifest, self.folder / "run", ROOT, phase="debug")
        self.assertEqual(partitions, ["train", "validation", "train", "validation"])

    def test_resume_rejects_changed_method_recipe_and_runtime(self):
        """Resuming an incomplete run must not silently restart or change experimental conditions."""
        config = debug_config()
        directory = self.folder / "run"
        train(config, self.manifest, directory, ROOT, phase="debug", stop_after=1)
        for replacement in (
            replace(config, gate_tau=0.3),
            replace(config, method="erm"),
            replace(config, seed=1),
        ):
            with self.assertRaises(ValueError):
                train(replacement, self.manifest, directory, ROOT, phase="debug", resume=True)
        contract = read_json(directory / "contract.json")
        contract["runtime"]["python"] = "changed"
        atomic_json(directory / "contract.json", contract)
        with self.assertRaises(ValueError):
            train(config, self.manifest, directory, ROOT, phase="debug", resume=True)

    def test_completion_rejects_changed_final_weights_and_history(self):
        """The final seal binds both the selected tensor file and validation evidence."""
        directory = self.folder / "run"
        config = debug_config()
        train(config, self.manifest, directory, ROOT, phase="debug")
        history = directory / "history.json"
        original = history.read_bytes()
        history.write_text("[]")
        with self.assertRaises(ValueError):
            verify_run(directory, ROOT)
        history.write_bytes(original)
        with (directory / "final.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            verify_run(directory, ROOT)

    def test_final_test_requires_study_seal_and_cannot_be_used_for_development(self):
        """No test loader exists until the study authorization is checked."""
        directory = self.folder / "run"
        train(debug_config(), self.manifest, directory, ROOT, phase="development")
        seal = {
            "purpose": "final_evaluation_after_all_prespecified_training",
            "runs": {str(directory.resolve()): file_hash(directory / "completion.json")},
            "source_sha256": read_json(directory / "contract.json")["source_sha256"],
        }
        with patch(
            "clvision.engine.make_loader", side_effect=AssertionError("test loader must not open")
        ):
            with self.assertRaises(ValueError):
                evaluate_run(self.manifest, directory, ROOT, {})
            with self.assertRaises(ValueError):
                evaluate_run(self.manifest, directory, ROOT, seal)

    def test_completed_run_reuse_and_clean_prediction_integrity(self):
        """Reusing completed training and evaluations preserves their committed artifacts."""
        config = debug_config()
        directory = self.folder / "run"
        train(config, self.manifest, directory, ROOT, phase="debug")
        completion = file_hash(directory / "completion.json")
        contract = read_json(directory / "contract.json")
        seal = {
            "purpose": "final_evaluation_after_all_prespecified_training",
            "source_sha256": contract["source_sha256"],
            "runs": {str(directory.resolve()): completion},
        }
        first = evaluate_run(self.manifest, directory, ROOT, seal)
        self.assertEqual(first, evaluate_run(self.manifest, directory, ROOT, seal))
        train(config, self.manifest, directory, ROOT, phase="debug", resume=True)
        self.assertEqual(completion, file_hash(directory / "completion.json"))
        with self.assertRaises(ValueError):
            train(
                replace(config, gate_tau=0.3),
                self.manifest,
                directory,
                ROOT,
                phase="debug",
                resume=True,
            )
        (directory / "test_predictions.pt").write_bytes(b"altered")
        with self.assertRaises(ValueError):
            evaluate_run(self.manifest, directory, ROOT, seal)

    def test_amp_backoff_restores_batchnorm_buffers_and_sgd_counts(self):
        """Real CPU GradScaler calibration retries must accept exactly one momentum update."""

        class HalfGradient(nn.Module):
            """Produce finite losses with overflowing scaled FP16 backward intermediates."""

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.register_buffer("calls", torch.tensor(0))

            def forward(self, images):
                self.calls.add_(1)
                return self.weight.to(torch.float16).float() * 2

        model = HalfGradient()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536)

        def objective(model, images, targets, config):
            return model(images), {}

        with patch("clvision.engine.training_objective", side_effect=objective):
            diagnostics = optimize_batch(
                model, optimizer, scaler, torch.ones(1), torch.zeros(1), debug_config()
            )
        self.assertGreater(diagnostics["amp_overflow_retries"], 0)
        self.assertEqual(int(model.calls), 1)
        self.assertAlmostEqual(float(model.weight.detach()), 0.8, places=6)
        self.assertAlmostEqual(
            float(optimizer.state[model.weight]["momentum_buffer"]), 2.0, places=6
        )

    def test_nonfinite_forward_stops_before_any_sgd_update(self):
        """Loss scaling cannot repair a nonfinite forward calculation."""
        model = nn.Linear(1, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with patch(
            "clvision.engine.training_objective",
            return_value=(torch.tensor(float("nan"), requires_grad=True), {}),
        ):
            with self.assertRaises(FloatingPointError):
                optimize_batch(
                    model,
                    optimizer,
                    torch.amp.GradScaler("cpu", enabled=False),
                    torch.ones(1, 1),
                    torch.zeros(1),
                    debug_config(),
                )
        self.assertTrue(torch.equal(before, model.weight))

    def test_counterfactual_metrics_distinguish_shortcut_from_core(self):
        """A digit-only classifier fails reversed cues while a vehicle classifier is invariant."""
        labels = torch.tensor([0, 0, 1, 1])
        shortcut = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        reliable = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
        first, second = dominoes_metrics(shortcut, labels), dominoes_metrics(reliable, labels)
        self.assertAlmostEqual(first["accuracy_by_digit_agreement"]["0.95"], 0.95)
        self.assertAlmostEqual(first["accuracy_by_digit_agreement"]["0.5"], 0.5)
        self.assertAlmostEqual(first["accuracy_by_digit_agreement"]["0.05"], 0.05)
        self.assertEqual(first["worst_group_accuracy"], 0)
        self.assertEqual(second["worst_group_accuracy"], 1)
        self.assertEqual(second["digit_counterfactual_prediction_flip_rate"], 0)

    def test_all_corruptions_evaluate_matching_clean_checkpoints(self):
        """The full75-cell evaluation resumes cells without retraining or test adaptation."""
        from clvision.config import CORRUPTIONS

        directory = self.folder / "run"
        config = debug_config("erm")
        train(config, self.manifest, directory, ROOT, phase="debug")
        contract = read_json(directory / "contract.json")
        seal = {
            "purpose": "final_evaluation_after_all_prespecified_training",
            "source_sha256": contract["source_sha256"],
            "runs": {str(directory.resolve()): file_hash(directory / "completion.json")},
        }
        images = np.zeros((60, 32, 32, 3), dtype=np.uint8)
        descriptor = save_array(self.manifest["root"], "fixture/corruption.npy", images)
        corruptions = {
            "root": self.manifest["root"],
            "dataset_id": "test-corruption-grid",
            "clean_dataset_id": self.manifest["dataset_id"],
            "corruptions": {name: descriptor for name in CORRUPTIONS},
        }
        first = evaluate_corruptions(corruptions, self.manifest, directory, ROOT, seal)
        second = evaluate_corruptions(corruptions, self.manifest, directory, ROOT, seal)
        self.assertEqual(first, second)
        self.assertEqual(len(first["cells"]), 75)
        self.assertTrue(all(item["n"] == 12 for item in first["cells"].values()))
        self.assertEqual(
            read_json(directory / "completion.json")["contract_hash"], canonical_hash(contract)
        )
