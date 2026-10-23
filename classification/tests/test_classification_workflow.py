"""Registered scope, development freezing, study ordering, and failed-preflight regression tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clvision.config import CORRUPTIONS, MAIN_METHODS, MECHANISM_METHODS
from clvision.data import ImageDataset, make_fixture
from clvision.profile import profile_dataset, verify_profile
from clvision.workflow import (
    base_recipe,
    candidate_configs,
    choose_candidates,
    corruption_summary,
    freeze_plan,
    run_study,
    seal_study,
)

from cue_routing.io import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[2]


class ClassificationWorkflowTests(unittest.TestCase):
    """Test experiment decisions without replacing them with implementation-shaped assertions."""

    def setUp(self):
        """Create all three small fixture manifests under one isolated root."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.manifests = {
            name: make_fixture(self.folder / "data", name)
            for name in ("cifar100", "tiny_imagenet", "dominoes")
        }

    def tearDown(self):
        """Remove only the isolated verification artifacts."""
        self.temporary.cleanup()

    def test_development_candidates_and_validation_tiebreak(self):
        """Give each intervention the declared budget and deterministic score-based selection."""
        base = base_recipe("cifar100", {"debug": True, "workers": 0})
        candidates = candidate_configs(base)
        self.assertEqual(len(candidates), 9)
        rows = []
        for index, (name, config) in enumerate(candidates):
            rows.append(
                {
                    "name": name,
                    "config": config.to_dict(),
                    "validation": {"accuracy": 0.5, "nll": 2.0 if index % 3 != 1 else 1.0},
                }
            )
        selected = choose_candidates(rows)
        self.assertEqual(selected["cue"]["gate_tau"], 0.15)
        self.assertEqual(selected["fixed_global_local"]["fixed_lambda"], 0.25)
        self.assertEqual(selected["zipfls_dense2"]["zipf_lambda"], 0.1)
        with self.assertRaises(ValueError):
            choose_candidates(rows[:-1])

    def test_saved_labels_are_checked_against_canonical_dataset_targets(self):
        """Canonical labels cover image records and counterfactual vehicle pairs alike."""
        for manifest in self.manifests.values():
            for split in ("train", "validation", "test"):
                dataset = ImageDataset(manifest, split)
                self.assertEqual(
                    [dataset.sample_target(index) for index in range(len(dataset))],
                    [dataset[index][1] for index in range(len(dataset))],
                )

    def test_corruption_summary_rejects_partial_stale_and_inconsistent_results(self):
        """Require every registered cell, the right checkpoint, full counts, and a supported mean."""
        artifact = {
            "contract_hash": "contract",
            "study_seal_sha256": "seal",
            "dataset_id": "data",
            "cells": {
                f"{name}/{severity}": {"n": 10000, "accuracy": severity / 10}
                for name in CORRUPTIONS
                for severity in range(1, 6)
            },
            "mean_corruption_accuracy": 0.3,
        }
        result = corruption_summary(artifact, "contract", "seal", "data", 10000)
        self.assertAlmostEqual(result["mean_accuracy"], 0.3)
        self.assertAlmostEqual(result["mean_error"], 0.7)
        self.assertEqual(len(result["by_corruption"]), 15)
        self.assertAlmostEqual(result["by_severity"]["5"], 0.5)
        for update in (
            {"contract_hash": "another checkpoint"},
            {"dataset_id": "another data release"},
            {"cells": dict(list(artifact["cells"].items())[:-1])},
            {"cells": {key: {**value, "n": 9999} for key, value in artifact["cells"].items()}},
            {"mean_corruption_accuracy": 0.99},
        ):
            with self.assertRaises(ValueError):
                corruption_summary({**artifact, **update}, "contract", "seal", "data", 10000)

    def test_production_scope_is_twenty_main_and_twelve_mechanism_runs(self):
        """The requested two-seed pilot must retain every declared method and dataset."""
        folder = self.folder / "study"
        folder.mkdir()
        plan = freeze_plan(folder, self.manifests, ROOT, {"debug": False, "workers": 4})
        self.assertEqual(plan["main_seeds"], [0, 1])
        self.assertEqual(plan["mechanism_seeds"], [0, 1])
        self.assertEqual(
            len(plan["main_seeds"]) * len(MAIN_METHODS) * len(plan["natural_datasets"]), 20
        )
        self.assertEqual(len(plan["mechanism_seeds"]) * len(MECHANISM_METHODS), 12)
        self.assertEqual(
            len(candidate_configs(base_recipe("cifar100", {"debug": False, "workers": 4}))), 9
        )
        with self.assertRaises(ValueError):
            freeze_plan(folder, self.manifests, ROOT, {"debug": True, "workers": 0})
        atomic_json(
            folder / "plan.json",
            {**plan, "main_seeds": [0, 1, 2, 3, 4], "mechanism_seeds": [0, 1, 2]},
        )
        with self.assertRaises(ValueError):
            freeze_plan(folder, self.manifests, ROOT, {"debug": False, "workers": 4})

    def test_incomplete_scope_cannot_grant_test_access(self):
        """A partial collection of completed runs never authorizes final evaluation."""
        folder = self.folder / "study"
        folder.mkdir()
        plan = freeze_plan(folder, self.manifests, ROOT, {"debug": True, "workers": 0})
        with self.assertRaises(ValueError):
            seal_study(folder, ROOT, plan, [], self.manifests)
        self.assertFalse((folder / "study_seal.json").exists())

    def test_every_training_run_precedes_any_final_evaluation(self):
        """Audit real workflow ordering and separate development from held-out predictions."""
        events, saved = [], {}

        def fake_train(
            config, manifest, directory, root, phase="main", selection_id="", resume=False
        ):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            events.append(("train", phase, config.dataset))
            saved[str(directory)] = (config, manifest, phase)
            atomic_json(
                directory / "completion.json", {"method": config.method, "seed": config.seed}
            )

        def fake_verify(directory, root, *args, **kwargs):
            config, manifest, phase = saved[str(directory)]
            return {"phase": phase}, {}, {"final_validation": {"accuracy": 0.25, "nll": 3.0}}

        def fake_evaluate(manifest, directory, root, seal):
            events.append(("evaluate", "final", manifest["dataset"]))
            self.assertEqual(sum(event[0] == "train" for event in events), 34)
            return {}

        with (
            patch("clvision.workflow.train", side_effect=fake_train),
            patch("clvision.workflow.verify_run", side_effect=fake_verify),
            patch("clvision.workflow.evaluate_run", side_effect=fake_evaluate),
            patch("clvision.workflow.report", return_value={"passed": True}),
        ):
            result = run_study(
                ROOT, self.folder / "data", self.folder / "study", debug=True, workers=0
            )
        self.assertTrue(result["passed"])
        self.assertEqual(sum(event[1] == "development" for event in events), 18)
        self.assertEqual(sum(event[0] == "evaluate" for event in events), 16)
        first_mechanism = next(
            index for index, event in enumerate(events) if event[1] == "mechanism"
        )
        first_nine = events[:first_mechanism]
        self.assertEqual(len(first_nine), 9)
        self.assertTrue(all(event[1:] == ("development", "cifar100") for event in first_nine))

    def test_no_cuda_profile_never_claims_gpu_fit(self):
        """A CUDA-capable software installation is not a measured full-batch pass."""
        base = base_recipe("cifar100", {"debug": False, "workers": 0})
        path = self.folder / "cuda.json"
        with patch("clvision.profile.torch.cuda.is_available", return_value=False):
            result = profile_dataset(ROOT, self.manifests["cifar100"], base, path)
        self.assertFalse(result["measured"])
        self.assertFalse(result["passes_budget"])
        with self.assertRaises(ValueError):
            verify_profile(result, ROOT, self.manifests["cifar100"], base)

    def test_failed_preflight_prevents_development_and_study_creation(self):
        """Any failed shape/device preflight must stop before expensive experiment training."""
        fixture = dict(self.manifests["cifar100"])
        with (
            patch("clvision.workflow.load_manifest", return_value=fixture),
            patch(
                "clvision.workflow.profile_dataset",
                return_value={"measured": False, "passes_budget": False},
            ),
            patch("clvision.workflow.develop", side_effect=AssertionError("must not train")),
        ):
            with self.assertRaises(ValueError):
                run_study(
                    ROOT, self.folder / "data", self.folder / "blocked", debug=False, workers=0
                )
        self.assertFalse((self.folder / "blocked").exists())

    def test_failed_driver_command_stops_later_stages(self):
        """The documented preparation driver must not run after a failed test command."""
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location(
            "classification_driver", ROOT / "classification/run.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (
            patch("sys.argv", ["run.py", "prepare"]),
            patch.object(
                module.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "check")
            ) as calls,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                module.main()
        self.assertEqual(calls.call_count, 1)

    def test_frozen_selection_detects_changed_candidate_scores(self):
        """Saved validation evidence cannot be rewritten after choices are frozen."""
        from clvision.workflow import develop

        base = base_recipe("cifar100", {"debug": True, "workers": 0})
        directory = self.folder / "development"
        saved = {}

        def fake_train(config, manifest, run_dir, root, **kwargs):
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            atomic_json(
                Path(run_dir) / "completion.json",
                {"tau": config.gate_tau, "lambda": config.fixed_lambda, "zipf": config.zipf_lambda},
            )
            saved[str(run_dir)] = config

        def fake_verify(run_dir, root, *args):
            return {"phase": "development"}, {}, {"final_validation": {"accuracy": 0.5, "nll": 1.0}}

        with (
            patch("clvision.workflow.train", side_effect=fake_train),
            patch("clvision.workflow.verify_run", side_effect=fake_verify),
        ):
            first = develop(ROOT, self.manifests["cifar100"], directory, base)
            self.assertEqual(first, develop(ROOT, self.manifests["cifar100"], directory, base))
            value = read_json(directory / "selection.json")
            value["candidates"][0]["validation"]["accuracy"] = 0.8
            atomic_json(directory / "selection.json", value)
            with self.assertRaises(ValueError):
                develop(ROOT, self.manifests["cifar100"], directory, base)
