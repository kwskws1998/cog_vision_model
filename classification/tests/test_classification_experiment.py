"""Test split isolation, matched initialization, timing admission and validation-only promotion."""

import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from clvision import data
from clvision.config import Config, learning_rate_for_epoch
from clvision.display import DisplayOptions, display_options
from clvision.engine import make_model
from clvision.experiment import Comparison
from clvision.experiment_artifacts import split_audit
from clvision.experiment_runtime import (
    admit_candidate,
    preflight_search,
)
from clvision.experiment_settings import Budget, Search, deadline_time

from cue_routing.io import atomic_json, file_hash, read_json, seed_everything, state_hash

ROOT = Path(__file__).resolve().parents[2]


def validate_settings(value):
    """Adapt test inputs to the public Python recipe types."""
    base = Config.from_dict(value["training"])
    return base, Search(value["candidates"]).resolve(base)


def run_comparison(
    root,
    data_root,
    directory,
    settings,
    finish_by,
    resume=False,
    stages=("preflight", "tune", "baseline", "evaluate"),
):
    """Exercise the public context-managed API using caller-provided settings."""
    with Comparison(
        root=root,
        data_root=data_root,
        directory=directory,
        config=Config.from_dict(settings["training"]),
        search=Search(settings["candidates"]),
        budget=Budget(
            finish_by=finish_by,
            reserve_minutes=settings["reserve_minutes"],
            time_safety_factor=settings["time_safety_factor"],
        ),
        resume=resume,
    ) as experiment:
        return experiment.run(stages=stages)


class ExperimentTests(unittest.TestCase):
    """Exercise actual model training and final predictions without requiring a CUDA device."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.preset = {
            "training": {
                "method": "erm",
                "activation": "preact",
                "initialization": "msra",
                "normalization": "zipfls",
                "precision": "fp32",
            },
            "candidates": [{"gate_tau": 0.15}, {"gate_tau": 0.05}],
            "reserve_minutes": 0,
            "time_safety_factor": 1.25,
        }
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(2)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(display_options(DisplayOptions(progress="off")))

    def debug_settings(self):
        """Use a tiny explicit fixture run while preserving the real ResNet and split boundaries."""
        value = copy.deepcopy(self.preset)
        value["training"].update(
            debug=True,
            device="cpu",
            epochs=2,
            milestones=[1],
            workers=0,
            batch_size=2,
            eval_batch_size=4,
            debug_max_batches=1,
        )
        value["candidates"] = value["candidates"][:2]
        value["reserve_minutes"] = 0
        return value

    def test_editable_recipe_and_same_seed_for_every_candidate(self):
        """Caller changes common settings once; CUE-only overrides cannot secretly change them."""
        base, candidates = validate_settings(self.preset)
        self.assertEqual((base.seed, base.batch_size, base.learning_rate), (0, 128, 0.1))
        self.assertEqual({c.seed for c in candidates}, {0})
        self.assertEqual(base.initialization, "msra")
        self.assertEqual(base.normalization, "zipfls")
        for epoch, expected in ((100, 0.1), (101, 0.01), (150, 0.01), (151, 0.001)):
            self.assertAlmostEqual(learning_rate_for_epoch(base, epoch), expected)
        changed = copy.deepcopy(self.preset)
        changed["candidates"][0]["learning_rate"] = 0.8
        with self.assertRaises(ValueError):
            validate_settings(changed)
        changed = copy.deepcopy(self.preset)
        changed["training"]["batch_size"] = 1024
        modified, _ = validate_settings(changed)
        self.assertEqual(modified.batch_size, 1024)

    def test_three_splits_and_no_test_arrays_during_training_or_validation(self):
        """Raise immediately if preprocessing or manifest verification opens a deferred test array."""
        manifest = data.make_fixture(self.folder / "data")
        original = data.read_array
        opened = []

        def guarded(root, descriptor):
            """Permit train-source files only before the explicit final-test phase."""
            opened.append(descriptor["path"])
            self.assertNotIn("_test_", descriptor["path"])
            return original(root, descriptor)

        with patch("clvision.data.read_array", side_effect=guarded):
            loaded = data.load_manifest(
                self.folder / "data/manifests/cifar100.json", defer_test=True
            )
            for split in ("train", "validation"):
                dataset = data.ImageDataset(loaded, split)
                dataset[0]
        self.assertTrue(opened)
        self.assertTrue(split_audit(loaded)["pairwise_disjoint"])
        self.assertEqual(split_audit(loaded)["counts"], {"train": 8, "validation": 4, "test": 12})
        invalid = copy.deepcopy(manifest)
        invalid["splits"]["validation"][0] = invalid["splits"]["train"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_audit(invalid)
        test_file = self.folder / "data" / manifest["arrays"]["test_images"]["path"]
        with test_file.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "changed"):
            data.load_manifest(self.folder / "data/manifests/cifar100.json")

    def test_msra_is_explicit_and_erm_cue_initial_weights_match(self):
        """Verify Conv variance and paired exact weights while preserving the old default initializer."""
        config = Config(
            activation="preact", initialization="msra", device="cpu", precision="fp32", debug=True
        )
        seed_everything(0)
        first = make_model(replace(config, method="erm"))
        seed_everything(0)
        second = make_model(replace(config, method="cue"))
        self.assertEqual(first.shared_state_hash(), second.shared_state_hash())
        weight = first.backbone[0].weight
        self.assertAlmostEqual(float(weight.detach().var()), 2 / (64 * 3 * 3), delta=0.0005)
        seed_everything(0)
        old = make_model(replace(config, initialization="reference"))
        self.assertNotEqual(first.shared_state_hash(), old.shared_state_hash())

    def test_normalization_matches_explicit_author_constants_in_every_split(self):
        """Check normalized pixels independently instead of trusting a printed configuration field."""
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        actual = data.transform_image(image, "cifar100", normalization="zipfls")
        expected = (
            torch.tensor(128 / 255) - torch.tensor([0.4914, 0.4822, 0.4465])
        ) / torch.tensor([0.2023, 0.1994, 0.2010])
        torch.testing.assert_close(actual[:, 16, 16], expected)
        self.assertFalse(torch.equal(actual, data.transform_image(image, "cifar100")))
        manifest = data.make_fixture(self.folder / "data")
        config, _ = validate_settings(self.debug_settings())
        for split in ("train", "validation", "test"):
            self.assertEqual(
                data.make_loader(manifest, config, split).dataset.normalization, "zipfls"
            )

    def test_deadline_boundary_and_timezone(self):
        """Keep the reserve and uncertainty margin; insufficient time never shortens epochs."""
        self.assertTrue(admit_candidate(0, 100, 40, 0.5, 1.25))
        self.assertFalse(admit_candidate(0, 79, 40, 0.5, 1.25))
        self.assertEqual(
            deadline_time("2026-10-24T10:30:00+09:00"), deadline_time("2026-10-24T01:30:00+00:00")
        )
        with self.assertRaises(ValueError):
            deadline_time("2026-10-24T10:30:00")

    def test_end_to_end_closed_search_two_tests_no_retraining_and_resume(self):
        """Deny every test-array read before closure; recompute predictions and preserve winner weights."""
        settings = self.debug_settings()
        data_root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(data_root)
        original = data.read_array
        test_reads = []
        from clvision.experiment import train as actual_train

        training_order = []

        def ordered_train(config, manifest, run_dir, root, **kwargs):
            """Assert real CUE training completes and selection freezes before any ERM training."""
            training_order.append(config.method)
            if config.method == "cue":
                self.assertFalse((directory / "main/seed0_erm").exists())
            else:
                self.assertTrue((directory / "selection.json").exists())
                self.assertEqual(len(list((directory / "development").rglob("completion.json"))), 2)
            return actual_train(config, manifest, run_dir, root, **kwargs)

        def guarded(root, descriptor):
            """Require the complete search, frozen winner and training seal before test data loading."""
            if "_test_" in descriptor["path"]:
                self.assertTrue((directory / "selection.json").is_file())
                self.assertTrue((directory / "study_seal.json").is_file())
                self.assertEqual(len(list((directory / "development").rglob("completion.json"))), 2)
                self.assertFalse(list((directory / "development").rglob("test_access.json")))
                test_reads.append(descriptor["path"])
            return original(root, descriptor)

        with (
            patch("clvision.data.read_array", side_effect=guarded),
            patch("clvision.experiment.train", side_effect=ordered_train),
        ):
            result = run_comparison(
                ROOT, data_root, directory, settings, "2099-01-01T10:30:00+09:00"
            )
            self.assertEqual(training_order, ["cue", "cue", "erm"])
            before = {str(p): file_hash(p) for p in directory.rglob("completion.json")}
            again = run_comparison(
                ROOT, data_root, directory, settings, "2099-01-01T10:30:00+09:00", resume=True
            )
        self.assertTrue(test_reads)
        self.assertEqual(before, {str(p): file_hash(p) for p in directory.rglob("completion.json")})
        self.assertEqual(result["test_metrics"], again["test_metrics"])
        self.assertFalse(result["benchmark_evidence"])
        self.assertEqual(set(result["test_metrics"]), {"erm", "cue"})
        self.assertEqual(len(list(directory.rglob("test_access.json"))), 2)
        for method in ("erm", "cue"):
            folder = directory / "main" / f"seed0_{method}"
            self.assertEqual(len(read_json(folder / "test_access.json")), 1)
            prediction = torch.load(folder / "test_predictions.pt", weights_only=True)
            measured = (
                prediction["logits"].argmax(1).eq(prediction["targets"]).double().mean().item()
            )
            self.assertAlmostEqual(measured, result["test_metrics"][method]["accuracy"])
        chosen = result["selection"]["selected"]["name"]
        source = torch.load(directory / "development" / chosen / "final.pt", weights_only=True)
        final = torch.load(directory / "main/seed0_cue/final.pt", weights_only=True)
        self.assertEqual(state_hash(source["model"]), state_hash(final["model"]))
        frozen = read_json(directory / "selection.json")
        atomic_json(directory / "selection.json", {**frozen, "selected": {}})
        with self.assertRaisesRegex(ValueError, "Frozen validation selection"):
            run_comparison(
                ROOT, data_root, directory, settings, "2099-01-01T10:30:00+09:00", resume=True
            )

    def test_time_budget_skips_candidates_and_never_tests_unselected_runs(self):
        """Force timing admission to accept only the first CUE candidate and close the rest."""
        data_root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(data_root)
        with patch(
            "clvision.experiment.admit_candidate", side_effect=[True, True, False]
        ) as admission:
            result = run_comparison(
                ROOT, data_root, directory, self.debug_settings(), "2099-01-01T10:30:00+09:00"
            )
        self.assertEqual(len(result["selection"]["candidates"]), 1)
        self.assertEqual(result["selection"]["search"]["decisions"][1]["action"], "skip_budget")
        self.assertFalse((directory / "development/cue1").exists())
        self.assertEqual(len(list(directory.rglob("test_access.json"))), 2)
        for decision, call in zip(
            result["selection"]["search"]["decisions"], admission.call_args_list[1:]
        ):
            self.assertGreater(decision["reserved_erm_seconds"], 0)
            self.assertAlmostEqual(
                call.args[2], decision["estimated_seconds"] + decision["reserved_erm_seconds"]
            )

    def test_every_exact_cue_candidate_is_preflighted_before_erm(self):
        """Check all candidate-specific configs reach CUDA profiling before the ERM measurement."""
        base, candidates = validate_settings(self.preset)
        calls = []

        def profiler(root, manifest, config, destination, methods, steps):
            """Return only the measured identities needed to exercise scheduling without CUDA."""
            calls.append((config, destination, methods))
            return {
                "active": {"config": config.to_dict()},
                "optimizer_steps_per_method": steps,
                "methods": {
                    config.method: {
                        "initial_shared_model_sha256": "same-initial-weights",
                        "first_augmented_batch_sha256": "same-training-batch",
                    }
                },
            }

        with (
            patch("clvision.experiment_runtime.profile_dataset", side_effect=profiler),
            patch("clvision.experiment_runtime.verify_profile") as verify,
        ):
            erm, cues = preflight_search(ROOT, self.folder, {}, base, candidates)
        self.assertEqual([row[0] for row in calls], candidates + [base])
        self.assertEqual([row[2] for row in calls], [("cue",)] * len(candidates) + [("erm",)])
        self.assertEqual(verify.call_count, len(candidates) + 1)
        self.assertEqual(len(cues), len(candidates))
        self.assertIn("erm", erm["methods"])
        cached_path = self.folder / "preflight/cue0.json"
        atomic_json(cached_path, cues[1])
        with (
            patch("clvision.experiment_runtime.profile_dataset") as unused,
            patch("clvision.experiment_runtime.verify_profile"),
            self.assertRaisesRegex(ValueError, "exact registered candidate config"),
        ):
            preflight_search(ROOT, self.folder, {}, base, candidates)
        unused.assert_not_called()
        cached_path.unlink()
        with (
            patch(
                "clvision.experiment_runtime.profile_dataset",
                side_effect=[cues[0], RuntimeError("candidate1 failed")],
            ) as failed,
            patch("clvision.experiment_runtime.verify_profile"),
            self.assertRaisesRegex(RuntimeError, "candidate1 failed"),
        ):
            preflight_search(ROOT, self.folder, {}, base, candidates)
        self.assertEqual([call.args[2].method for call in failed.call_args_list], ["cue", "cue"])

    def test_failed_candidate_preflight_prevents_all_long_training_and_testing(self):
        """A candidate failure must surface before spending hours on either full training arm."""
        data_root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(data_root)
        with (
            patch(
                "clvision.experiment.preflight_search",
                side_effect=RuntimeError("candidate preflight failed"),
            ),
            patch("clvision.experiment.train") as training,
            self.assertRaisesRegex(RuntimeError, "candidate preflight failed"),
        ):
            run_comparison(
                ROOT, data_root, directory, self.debug_settings(), "2099-01-01T10:30:00+09:00"
            )
        training.assert_not_called()
        self.assertFalse(list(directory.rglob("test_access.json")))

    def test_insufficient_initial_budget_never_trains_or_tests(self):
        """An expired or too-small budget must not launch a long baseline anyway."""
        data_root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(data_root)
        with (
            patch("clvision.experiment.train") as training,
            self.assertRaisesRegex(RuntimeError, "cannot fit"),
        ):
            run_comparison(
                ROOT, data_root, directory, self.debug_settings(), "2000-01-01T10:30:00+09:00"
            )
        training.assert_not_called()
        self.assertFalse(list(directory.rglob("test_access.json")))

    def test_baseline_first_uses_same_engine_and_no_longer_reserves_erm(self):
        """Swap only stage order, change common inputs, and exercise actual training with no deadline."""
        from clvision.experiment import train as actual_train

        settings = self.debug_settings()
        settings["training"].update(
            activation="postact",
            batch_size=4,
            eval_batch_size=3,
            learning_rate=0.02,
            momentum=0.8,
            weight_decay=0.002,
            initialization="reference",
            normalization="dataset",
        )
        root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(root)
        calls = []

        def record(config, *args, **kwargs):
            """Observe real model training rather than replacing the training result."""
            calls.append(config.method)
            self.assertEqual(config.learning_rate, 0.02)
            self.assertEqual(config.batch_size, 4)
            return actual_train(config, *args, **kwargs)

        with patch("clvision.experiment.train", side_effect=record):
            output = run_comparison(
                ROOT,
                root,
                directory,
                settings,
                None,
                stages=("preflight", "baseline", "tune", "evaluate"),
            )
        self.assertEqual(calls, ["erm", "cue", "cue"])
        self.assertIsNone(output["deadline_met"])
        self.assertTrue(
            all(
                row["reserved_erm_seconds"] == 0
                for row in output["selection"]["search"]["decisions"]
            )
        )
        with self.assertRaisesRegex(ValueError, "Execution order changed"):
            run_comparison(ROOT, root, directory, settings, None, resume=True)

    def test_direct_stages_reject_early_evaluation_and_training_without_preflight(self):
        """Call public stages individually and enforce only scientific dependency boundaries."""
        settings = self.debug_settings()
        root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(root)
        base, _ = validate_settings(settings)
        with Comparison(
            root=ROOT,
            data_root=root,
            directory=directory,
            config=base,
            search=Search([{}]),
            budget=Budget(),
        ) as experiment:
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                experiment.baseline()
            with self.assertRaisesRegex(RuntimeError, "test remains sealed"):
                experiment.evaluate()
            experiment.preflight()
            experiment.tune()
            with self.assertRaisesRegex(RuntimeError, "test remains sealed"):
                experiment.evaluate()
            experiment.baseline()
            output = experiment.evaluate()
        self.assertEqual(len(output["selection"]["candidates"]), 1)
        self.assertEqual(len(list(directory.rglob("test_access.json"))), 2)

    def test_recipe_changes_rejected_on_resume_before_any_stage_runs(self):
        """Keep existing run identity strict while allowing different settings in a new directory."""
        settings = self.debug_settings()
        root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(root)
        run_comparison(ROOT, root, directory, settings, None)
        settings["training"]["learning_rate"] = 0.05
        with patch("clvision.experiment.preflight_search") as profile:
            with self.assertRaisesRegex(ValueError, "Study scope or source changed"):
                run_comparison(ROOT, root, directory, settings, None, resume=True)
        profile.assert_not_called()

    def test_numeric_representation_change_cannot_overwrite_frozen_plan(self):
        """Python-equal JSON values must not change a saved plan hash before resume rejects them."""
        settings = self.debug_settings()
        root, directory = self.folder / "data", self.folder / "study"
        data.make_fixture(root)
        config, _ = validate_settings(settings)
        with Comparison(
            root=ROOT,
            data_root=root,
            directory=directory,
            config=config,
            search=Search([{}]),
            budget=Budget(reserve_minutes=20),
        ):
            pass
        before = file_hash(directory / "plan.json")
        with self.assertRaisesRegex(ValueError, "Study scope or source changed"):
            with Comparison(
                root=ROOT,
                data_root=root,
                directory=directory,
                config=config,
                search=Search([{}]),
                budget=Budget(reserve_minutes=20.0),
                resume=True,
            ):
                pass
        self.assertEqual(file_hash(directory / "plan.json"), before)

    def test_tiny_record_test_files_remain_unopened_until_final_verification(self):
        """Handle TinyImageNet's file records without hashing held-out images during tuning."""
        from PIL import Image

        from cue_routing.io import file_hash

        root = self.folder / "tiny"
        root.mkdir()
        records, test_records = [], []
        for name, rows, count in (("train", records, 8), ("heldout", test_records, 4)):
            for index in range(count):
                file = root / f"{name}{index}.png"
                Image.fromarray(np.full((64, 64, 3), index, dtype=np.uint8)).save(file)
                rows.append([file.name, index, file_hash(file)])
        manifest = data.seal_manifest(
            root,
            {
                "dataset": "tiny_imagenet",
                "kind": "fixture",
                "classes": [str(i) for i in range(200)],
                "records": records,
                "test_records": test_records,
                "splits": {"train": list(range(6)), "validation": [6, 7], "test": list(range(4))},
                "shape": [64, 64],
                "split_policy": "official validation held out as test",
            },
        )
        original = data.file_hash

        def guarded(path):
            """Deny even read-only checksum access to the sealed held-out files."""
            self.assertFalse(Path(path).name.startswith("heldout"))
            return original(path)

        with patch("clvision.data.file_hash", side_effect=guarded):
            loaded = data.load_manifest(root / "manifests/tiny_imagenet.json", defer_test=True)
            for split in ("train", "validation"):
                data.ImageDataset(loaded, split)[0]
        self.assertEqual(split_audit(loaded)["source_split_policy"], manifest["split_policy"])
        (root / "heldout0.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Dataset image changed"):
            data.load_manifest(root / "manifests/tiny_imagenet.json")
