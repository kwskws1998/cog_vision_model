"""CUE-only scope, CLI precedence, GPU subset checks, and real sealed-report regression tests."""

import argparse
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.config import Config, learning_rate_for_epoch
from clvision.cue_settings import CueSettings, add_cue_arguments, load_cue_arguments
from clvision.cue_workflow import (
    cue_report,
    cue_specs,
    make_cue_plan,
    run_cue_study,
    seal_cue_study,
)
from clvision.data import make_fixture
from clvision.engine import train
from clvision.profile import profile_dataset, selected_methods, verify_profile
from clvision.progress import StudyProgress
from test_classification_training import debug_config, equal_states

from cue_routing.io import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]


class CueOnlyTests(unittest.TestCase):
    """Exercise the new workflow independently of a GPU or the old full-study matrix."""

    def setUp(self):
        """Isolate artifacts and use the shipped GPU preset as the production input."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.preset = read_json(ROOT / "classification/configs/cue_5090.json")

    def tearDown(self):
        """Remove only this test's temporary files."""
        self.temporary.cleanup()

    def test_default_scope_has_four_jobs_and_no_hidden_corruption_budget(self):
        """The launch must not grow back into a grid search or competitor/mechanism study."""
        settings = CueSettings.from_dict(self.preset)
        preview = settings.preview()
        self.assertEqual((preview["training_runs"], preview["training_epochs"]), (4, 800))
        manifests = {name: make_fixture(self.folder / "data", name) for name in settings.datasets}
        plan = make_cue_plan(ROOT, manifests, settings)
        jobs = StudyProgress(self.folder / "study", plan, manifests).jobs
        self.assertEqual(len(jobs), 4)
        self.assertTrue(
            all(job["method"] == "cue" and job["corruption_cells"] == 0 for job in jobs.values())
        )
        self.assertEqual(plan["development_trials"], {})
        for config in preview["resolved_recipes"].values():
            self.assertEqual(
                (config["batch_size"], config["eval_batch_size"], config["max_vram_gib"]),
                (1024, 2048, 28),
            )
            self.assertEqual((config["precision"], config["eval_precision"]), ("bf16", "fp32"))

    def test_cli_overrides_dataset_recipe_and_reduces_scope(self):
        """Explicit flags win over dataset JSON values and one dataset/seed is one job."""
        self.preset["dataset_overrides"] = {"tiny_imagenet": {"batch_size": 256}}
        path = self.folder / "inputs.json"
        atomic_json(path, self.preset)
        parser = argparse.ArgumentParser()
        add_cue_arguments(parser, ROOT)
        arguments = parser.parse_args(
            [
                "--config",
                str(path),
                "--datasets",
                "tiny_imagenet",
                "--seeds",
                "42",
                "--batch-size",
                "512",
                "--learning-rate",
                "0.4",
                "--gate-tau",
                "0.2",
            ]
        )
        settings = load_cue_arguments(arguments)
        config = settings.recipe("tiny_imagenet")
        self.assertEqual(
            (config.batch_size, config.learning_rate, config.gate_tau), (512, 0.4, 0.2)
        )
        self.assertEqual(settings.preview()["training_runs"], 1)

    def test_invalid_scope_and_inputs_fail_early(self):
        """Reject misspellings, competitor injection, duplicate jobs, and invalid warmup."""
        for value in (
            {"datasets": ["dominoes"]},
            {"datasets": ["cifar100", "cifar100"]},
            {"seeds": []},
            {"seeds": [0, 0]},
            {"seeds": [True]},
            {"training": {"method": "erm"}},
            {"training": {"workers": -1}},
            {"training": {"warmup_epochs": 200}},
            {"unknown": 1},
            {"training": {"max_vram_gib": float("nan")}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CueSettings.from_dict(value)
        Config(max_vram_gib=28).validate()
        Config(max_vram_gib=48).validate()
        with self.assertRaises(ValueError):
            Config(max_vram_gib=0).validate()

    def test_profile_checks_exact_subset_and_never_runs_competitors(self):
        """CUE-only preflight and verification must agree on the same selected intervention."""
        config = CueSettings.from_dict(self.preset).recipe("cifar100")
        evidence = {
            "vram": {"peak_reserved_gib": 10},
            "backbone_updated": True,
            "initial_shared_model_sha256": "same",
            "first_augmented_batch_sha256": "same",
            "training_images_per_second": 100,
            "evaluation_images_per_second": 100,
        }
        from clvision.profile import memory_signature

        evidence["signature"] = memory_signature(config)
        with (
            patch("clvision.profile.torch.cuda.is_available", return_value=True),
            patch("clvision.profile.environment", return_value={"gpu": "mock"}),
            patch("clvision.profile.profile_method", return_value=evidence) as measured,
        ):
            value = profile_dataset(
                ROOT, {"dataset_id": "fixture"}, config, self.folder / "gpu.json", methods=("cue",)
            )
            self.assertEqual(measured.call_count, 1)
            self.assertEqual(measured.call_args.args[1].method, "cue")
            verify_profile(value, ROOT, {"dataset_id": "fixture"}, config, methods=("cue",))
            with self.assertRaisesRegex(ValueError, "omitted"):
                verify_profile(value, ROOT, {"dataset_id": "fixture"}, config)
        for methods in ((), ("cue", "cue"), ("unknown",)):
            with self.assertRaises(ValueError):
                selected_methods(config, methods)

    def test_failed_preflight_never_starts_training(self):
        """An oversized or numerically failing CUDA recipe cannot silently shrink or start a study."""
        settings = CueSettings.from_dict({**self.preset, "datasets": ["cifar100"]})
        manifest = make_fixture(self.folder / "data", "cifar100")
        with (
            patch("clvision.cue_workflow.load_manifest", return_value=manifest),
            patch(
                "clvision.cue_workflow.profile_dataset",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
            patch("clvision.cue_workflow.train") as training,
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                run_cue_study(ROOT, self.folder / "data", self.folder / "run", settings)
            training.assert_not_called()

    def test_warmup_and_bitwise_resume_across_decay(self):
        """An interrupted warmup must replay SGD state and the scheduled learning rate exactly."""
        config = replace(
            debug_config(), epochs=3, milestones=(2,), warmup_epochs=2, learning_rate=0.8
        )
        self.assertEqual(
            [learning_rate_for_epoch(config, i) for i in (1, 2, 3)], [0.4, 0.8, 0.08000000000000002]
        )
        manifest = make_fixture(self.folder / "data", "cifar100")
        full, resumed = self.folder / "full", self.folder / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            train(config, manifest, full, ROOT, phase="debug")
            train(config, manifest, resumed, ROOT, phase="debug", stop_after=1)
            train(config, manifest, resumed, ROOT, phase="debug", resume=True)
        first, second = [
            torch.load(path / "last.pt", weights_only=True) for path in (full, resumed)
        ]
        for key in ("model", "optimizer", "scaler", "rng", "steps"):
            self.assertTrue(equal_states(first[key], second[key]), key)

    def test_real_cue_workflow_seals_before_test_and_reuses_validated_artifacts(self):
        """Real CPU execution covers both datasets, ordered final access, resume, and tamper rejection."""
        settings = CueSettings.from_dict({**self.preset, "debug": True, "seeds": [0]})
        data, directory = self.folder / "data", self.folder / "study"
        manifests = {name: make_fixture(data, name) for name in settings.datasets}
        from clvision.cue_workflow import evaluate_run

        def guarded(manifest, run_dir, root, seal):
            self.assertEqual(len(seal["runs"]), 2)
            self.assertTrue(
                all((Path(path) / "completion.json").is_file() for path in seal["runs"])
            )
            return evaluate_run(manifest, run_dir, root, seal)

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("clvision.cue_workflow.evaluate_run", side_effect=guarded),
        ):
            result = run_cue_study(ROOT, data, directory, settings)
            completions = {
                str(path): file_hash(path) for path in directory.rglob("completion.json")
            }
            self.assertEqual(result, run_cue_study(ROOT, data, directory, settings))
            self.assertEqual(
                completions,
                {str(path): file_hash(path) for path in directory.rglob("completion.json")},
            )
        progress = read_json(directory / "progress.json")
        self.assertEqual(
            (
                progress["status"],
                progress["total_training_runs"],
                progress["total_corruption_cells"],
            ),
            ("complete", 2, 0),
        )
        self.assertEqual(progress["estimated_remaining_seconds"], 0)
        self.assertFalse(result["benchmark_evidence"])
        self.assertFalse(result["paired_comparison"])
        self.assertEqual(len(result["datasets"]["cifar100"]["published_references"]), 9)
        self.assertEqual(result["datasets"]["cifar100"]["descriptive_differences"], [])
        plan, seal = read_json(directory / "plan.json"), read_json(directory / "study_seal.json")
        specs = cue_specs(directory, plan)
        with self.assertRaisesRegex(ValueError, "scope"):
            seal_cue_study(directory, ROOT, plan, specs[:-1], manifests)
        changed = CueSettings.from_dict({**settings.to_dict(), "seeds": [0, 1]})
        with self.assertRaisesRegex(ValueError, "scope or source"):
            run_cue_study(ROOT, data, directory, changed)
        run_dir = specs[0][0]
        predictions = torch.load(run_dir / "test_predictions.pt", weights_only=True)
        predictions["sample_ids"] = list(reversed(predictions["sample_ids"]))
        torch.save(predictions, run_dir / "test_predictions.pt")
        metrics = read_json(run_dir / "test_metrics.json")
        metrics["prediction_sha256"] = file_hash(run_dir / "test_predictions.pt")
        atomic_json(run_dir / "test_metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "sample order"):
            cue_report(directory, ROOT, plan, manifests, specs, seal)
