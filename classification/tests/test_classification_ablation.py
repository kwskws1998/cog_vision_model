"""Bounded tuning, matched ResNet controls, and sealed validation-only selection regressions."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.ablation_settings import (
    AblationSettings,
    add_ablation_arguments,
    load_ablation_arguments,
)
from clvision.ablation_workflow import (
    ablation_report,
    ablation_specs,
    make_ablation_plan,
    run_ablation,
    seal_ablation,
    winner,
)
from clvision.data import make_fixture
from clvision.progress import StudyProgress

from cue_routing.io import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]


class AblationTests(unittest.TestCase):
    """Exercise genuine training/evaluation with small fixtures and deterministic selection tests."""

    def setUp(self):
        """Allocate one isolated experiment directory for each regression."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.preset = read_json(ROOT / "classification/configs/ablation_5090.json")

    def tearDown(self):
        """Remove only the artifacts created by this test."""
        self.temporary.cleanup()

    def test_budget_and_conditional_recipes(self):
        """Default cost is ten CIFAR jobs, with optional datasets and fixed controls explicit."""
        settings = AblationSettings.from_dict(self.preset)
        preview = settings.preview()
        self.assertEqual(
            (preview["development_runs"], preview["main_runs"], preview["training_epochs"]),
            (6, 4, 2000),
        )
        self.assertEqual(
            AblationSettings.from_dict(
                {**self.preset, "datasets": ["cifar100", "tiny_imagenet"]}
            ).preview()["training_runs"],
            20,
        )
        self.assertEqual(
            AblationSettings.from_dict(
                {**self.preset, "methods": ["erm", "cue", "fixed_global_local"]}
            ).preview()["training_runs"],
            15,
        )
        self.assertEqual(
            AblationSettings.from_dict({**self.preset, "tune": False}).preview()["training_runs"], 4
        )
        trials = settings.trials("cifar100", shared_lr=0.4)
        self.assertEqual(
            [cfg.learning_rate for _, cfg in trials if cfg.method == "erm"], [0.1, 0.4, 0.8]
        )
        self.assertEqual([cfg.learning_rate for _, cfg in trials if cfg.method == "cue"], [0.4] * 3)
        manifest = make_fixture(self.folder / "data", "cifar100")
        manifests = {"cifar100": manifest}
        plan = make_ablation_plan(ROOT, manifests, settings)
        jobs = StudyProgress(self.folder / "study", plan, manifests).jobs
        self.assertEqual(len(jobs), 10)
        self.assertTrue(all(row["corruption_cells"] == 0 for row in jobs.values()))
        self.assertEqual(sum(row["clean_passes"] for row in jobs.values()), 4)

    def test_validation_selection_ignores_test_and_uses_registered_ties(self):
        """A high test value cannot win a validation-only search, including a complete tie."""
        rows = [
            {"name": "first", "validation": {"accuracy": 0.7, "nll": 1.0}, "test": 0.99},
            {"name": "second", "validation": {"accuracy": 0.8, "nll": 1.0}, "test": 0.1},
            {"name": "third", "validation": {"accuracy": 0.8, "nll": 0.9}, "test": 0.0},
        ]
        self.assertEqual(winner(rows)["name"], "third")
        self.assertEqual(winner([rows[2], {**rows[2], "name": "later"}])["name"], "third")
        with self.assertRaises(ValueError):
            winner([{**rows[0], "validation": {"accuracy": float("nan"), "nll": 1.0}}])

    def test_input_roles_and_cli_overrides(self):
        """Reject seed leakage and silently overridden tuning flags before any data access."""
        for updates in (
            {"development_seed": 0},
            {"development_seed": True},
            {"methods": ["cue"]},
            {"methods": ["erm", "cue", "cue"]},
            {"lr_candidates": []},
            {"tau_candidates": [float("nan")]},
            {"fixed_candidates": [2.0]},
            {"lr_candidates": [True]},
            {"lr_candidates": [0.1, 0.1]},
            {"tune": 1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                AblationSettings.from_dict({**self.preset, **updates})
        parser = argparse.ArgumentParser()
        add_ablation_arguments(parser, ROOT)
        self.assertEqual(
            load_ablation_arguments(
                parser.parse_args(["--lr-candidates", "0.2", "0.5"])
            ).lr_candidates,
            (0.2, 0.5),
        )
        with self.assertRaisesRegex(ValueError, "lr-candidates"):
            load_ablation_arguments(parser.parse_args(["--learning-rate", "0.2"]))
        fixed = load_ablation_arguments(
            parser.parse_args(["--no-tune", "--learning-rate", "0.2", "--gate-tau", "0.1"])
        )
        self.assertEqual(
            (fixed.recipe("cifar100").learning_rate, fixed.recipe("cifar100").gate_tau), (0.2, 0.1)
        )

    def test_real_tuning_matched_test_access_resume_and_tamper_rejection(self):
        """Actual CPU training verifies all development precedes four paired final evaluations."""
        settings = AblationSettings.from_dict(
            {
                **self.preset,
                "debug": True,
                "lr_candidates": [0.01, 0.02],
                "tau_candidates": [0.05, 0.15],
            }
        )
        data, directory = self.folder / "data", self.folder / "study"
        manifest = make_fixture(data, "cifar100")
        from clvision.ablation_workflow import evaluate_run

        def guarded(manifest, run_dir, root, seal):
            """Require completed trials, frozen selection, and every final model before testing."""
            self.assertEqual(len(list((directory / "development").rglob("completion.json"))), 4)
            self.assertFalse(list((directory / "development").rglob("test_access.json")))
            self.assertTrue((directory / "development/cifar100/selection.json").is_file())
            self.assertEqual(len(seal["runs"]), 4)
            self.assertTrue(
                all((Path(name) / "completion.json").is_file() for name in seal["runs"])
            )
            return evaluate_run(manifest, run_dir, root, seal)

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("clvision.ablation_workflow.evaluate_run", side_effect=guarded),
        ):
            result = run_ablation(ROOT, data, directory, settings)
            completions = {
                str(path): file_hash(path) for path in directory.rglob("completion.json")
            }
            self.assertEqual(result, run_ablation(ROOT, data, directory, settings))
            self.assertEqual(
                completions,
                {str(path): file_hash(path) for path in directory.rglob("completion.json")},
            )
        item = result["datasets"]["cifar100"]
        self.assertEqual(set(item["methods"]), {"erm", "cue"})
        self.assertFalse(result["benchmark_evidence"])
        self.assertTrue(result["paired_comparison"])
        self.assertFalse(result["formal_significance_claim"])
        selection = item["selection"]
        self.assertNotIn(settings.development_seed, settings.seeds)
        erm_winner = winner(
            [row for row in selection["candidates"] if row["config"]["method"] == "erm"]
        )
        self.assertEqual(
            selection["shared_config"]["learning_rate"], erm_winner["config"]["learning_rate"]
        )
        for row in selection["candidates"]:
            if row["config"]["method"] == "cue":
                self.assertEqual(
                    row["config"]["learning_rate"], selection["shared_config"]["learning_rate"]
                )
        progress = read_json(directory / "progress.json")
        self.assertEqual(
            (progress["total_training_runs"], progress["estimated_remaining_seconds"]), (8, 0)
        )
        plan = read_json(directory / "plan.json")
        specs = ablation_specs(directory, settings, {"cifar100": selection})
        for seed in settings.seeds:
            configs = [
                {key: value for key, value in spec[1].to_dict().items() if key != "method"}
                for spec in specs
                if spec[1].seed == seed
            ]
            self.assertEqual(configs[0], configs[1])
        with self.assertRaisesRegex(ValueError, "omitted"):
            seal_ablation(directory, ROOT, plan, specs[:-1], {"cifar100": manifest})
        selection_path = directory / "development/cifar100/selection.json"
        atomic_json(selection_path, {**selection, "selection_id": "changed"})
        with self.assertRaisesRegex(ValueError, "selection changed"):
            seal_ablation(directory, ROOT, plan, specs, {"cifar100": manifest})
        atomic_json(selection_path, selection)
        run_dir = specs[0][0]
        predictions = torch.load(run_dir / "test_predictions.pt", weights_only=True)
        predictions["sample_ids"] = list(reversed(predictions["sample_ids"]))
        torch.save(predictions, run_dir / "test_predictions.pt")
        metrics = read_json(run_dir / "test_metrics.json")
        metrics["prediction_sha256"] = file_hash(run_dir / "test_predictions.pt")
        atomic_json(run_dir / "test_metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "sample order"):
            ablation_report(
                directory,
                ROOT,
                plan,
                specs,
                {"cifar100": manifest},
                read_json(directory / "study_seal.json"),
            )


if __name__ == "__main__":
    unittest.main()
