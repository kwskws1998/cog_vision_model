"""Verify one fixed control's provenance, validation-only training, and resumable test boundary."""

import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from clvision import data
from clvision.config import Config
from clvision.display import DisplayOptions, display_options
from clvision.single_experiment import SingleRun

from cue_routing.io import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
MODULE = "clvision.single_experiment"


class SingleExperimentTests(unittest.TestCase):
    """Exercise the real fixed objective on CPU and isolate CUDA preflight contracts with mocks."""

    def setUp(self):
        """Keep generated arrays, experiment snapshots, checkpoints and results outside the source."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()
        self.data_root = self.folder / "data"
        self.manifest = data.make_fixture(self.data_root)
        self.directory = self.folder / "study"
        self.source = self.folder / "experiment.py"
        self.source.write_text("FIXED_LAMBDA = 0.5\n")
        self.config = Config(
            method="fixed_global_local",
            fixed_lambda=0.5,
            activation="preact",
            initialization="msra",
            normalization="zipfls",
            device="cpu",
            precision="fp32",
            eval_precision="fp32",
            debug=True,
            epochs=2,
            milestones=(1,),
            batch_size=2,
            eval_batch_size=4,
            workers=0,
            debug_max_batches=1,
        ).validate()
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(display_options(DisplayOptions(progress="off")))

    def experiment(self, **overrides):
        """Construct the public runner with explicit inputs shared by the test cases."""
        inputs = {
            "root": ROOT,
            "data_root": self.data_root,
            "directory": self.directory,
            "config": self.config,
            "experiment_source": self.source,
        }
        return SingleRun(**{**inputs, **overrides})

    def production_config(self):
        """Keep the archived CIFAR training recipe while avoiding actual CUDA execution in mocks."""
        return replace(
            self.config,
            device="cuda",
            debug=False,
            debug_max_batches=0,
            epochs=200,
            milestones=(100, 150),
            batch_size=128,
            eval_batch_size=128,
            workers=4,
        ).validate()

    def production_profile(self, config):
        """Supply only the preflight fields consumed by the runner after independent verification."""
        return {
            "active": {"config": config.to_dict()},
            "optimizer_steps_per_method": 10,
            "methods": {
                config.method: {"initial_shared_model_sha256": "a" * 64},
            },
        }

    def test_fixed_control_deferred_test_and_completed_resume(self):
        """Train the actual fixed gate, then ensure a completed resume preserves sealed predictions."""
        experiment = self.experiment()
        opened_test = []
        original_read = data.read_array

        def checked_read(root, item):
            """Require both completion and the study seal before any final-test array is decoded."""
            if "_test_" in item["path"]:
                self.assertTrue((experiment.run_dir / "completion.json").is_file())
                self.assertTrue((self.directory / "study_seal.json").is_file())
                opened_test.append(item["path"])
            return original_read(root, item)

        with patch("clvision.data.read_array", side_effect=checked_read):
            with experiment:
                self.assertEqual(opened_test, [])
                experiment.preflight()
                experiment.train()
                self.assertEqual(opened_test, [])
                result = experiment.evaluate()
        self.assertTrue(opened_test)
        self.assertFalse(result["benchmark_evidence"])
        self.assertEqual(result["final_validation"]["n"], 4)
        self.assertEqual(result["final_test"]["n"], 12)
        self.assertEqual(result["protocol"]["training_runs"], 1)
        self.assertFalse(result["protocol"]["tuning"])
        plan = read_json(self.directory / "plan.json")
        self.assertEqual(plan["main_methods"], ["fixed_global_local"])
        self.assertEqual(plan["main_seeds"], [0])
        self.assertEqual(plan["natural_datasets"], ["cifar100"])
        self.assertEqual(plan["protocol"]["config"]["fixed_lambda"], 0.5)
        for row in read_json(experiment.run_dir / "history.json"):
            diagnostics = row["training"]
            self.assertEqual(diagnostics["joint_fraction"], 0.5)
            self.assertEqual(diagnostics["gate_min"], 0.5)
            self.assertEqual(diagnostics["gate_max"], 0.5)
            self.assertEqual(diagnostics["gate_within_image_std"], 0.0)
        artifacts = [
            experiment.run_dir / name
            for name in ("completion.json", "final.pt", "test_predictions.pt", "test_metrics.json")
        ] + [self.directory / "study_seal.json"]
        hashes = {str(path): file_hash(path) for path in artifacts}
        access = read_json(experiment.run_dir / "test_access.json")
        self.assertEqual(len(access), 1)
        self.assertEqual(access[0]["kind"], "clean_test")
        with self.experiment(resume=True) as resumed:
            resumed.preflight()
            resumed.train()
            second = resumed.evaluate()
        self.assertEqual(second["final_test"], result["final_test"])
        self.assertEqual({str(path): file_hash(path) for path in artifacts}, hashes)
        self.assertEqual(read_json(experiment.run_dir / "test_access.json"), access)

    def test_early_stages_leave_test_arrays_unread(self):
        """Reject calls outside the context and training or evaluation before their prerequisites."""
        experiment = self.experiment()
        for stage in (experiment.preflight, experiment.train, experiment.evaluate):
            with self.assertRaises(RuntimeError):
                stage()
        original_read = data.read_array

        def checked_read(root, item):
            """Disallow final-test decoding throughout incomplete experiment attempts."""
            self.assertNotIn("_test_", item["path"])
            return original_read(root, item)

        with patch("clvision.data.read_array", side_effect=checked_read):
            with experiment:
                with self.assertRaises(RuntimeError):
                    experiment.train()
                with self.assertRaises(RuntimeError):
                    experiment.evaluate()
                experiment.preflight()
                with self.assertRaises(RuntimeError):
                    experiment.evaluate()
        self.assertFalse((experiment.run_dir / "test_access.json").exists())

    def test_dataset_and_initial_mismatch_prevent_training(self):
        """Reject historical-reference mismatches before the expensive training function is called."""
        with patch(f"{MODULE}.train") as training:
            with self.assertRaisesRegex(ValueError, "Dataset identity"):
                with self.experiment(expected_dataset_id="wrong"):
                    self.fail("Mismatched dataset must not enter the context")
            with self.experiment(expected_initial_weights="a" * 64) as experiment:
                with patch(f"{MODULE}.make_model") as model:
                    model.return_value.shared_state_hash.return_value = "b" * 64
                    with self.assertRaisesRegex(ValueError, "Initial model weights"):
                        experiment.preflight()
                with self.assertRaises(RuntimeError):
                    experiment.train()
            training.assert_not_called()

    def test_changed_resume_inputs_rejected_before_work(self):
        """Refuse a different fixed coefficient, script, or executable hash before any preflight."""
        with self.experiment():
            pass
        with patch(f"{MODULE}.profile_dataset") as profile, patch(f"{MODULE}.train") as training:
            with self.assertRaises(FileExistsError):
                with self.experiment():
                    self.fail("Existing study requires explicit resume")
            with self.assertRaisesRegex(ValueError, "recipe or source changed"):
                with self.experiment(config=replace(self.config, fixed_lambda=0.25), resume=True):
                    self.fail("Changed recipe must not enter the context")
            with patch(f"{MODULE}.source_hash", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "recipe or source changed"):
                    with self.experiment(resume=True):
                        self.fail("Changed executable source must not enter the context")
            self.source.write_text("FIXED_LAMBDA = 0.25\n")
            with self.assertRaisesRegex(ValueError, "recipe or source changed"):
                with self.experiment(resume=True):
                    self.fail("Changed experiment script must not enter the context")
            profile.assert_not_called()
            training.assert_not_called()
        with self.experiment(directory=self.folder / "fresh"):
            pass

    def test_source_mutation_inside_context_rejected_before_training(self):
        """Keep an edited live script from changing the meaning of an already frozen experiment."""
        with self.experiment() as experiment:
            experiment.preflight()
            self.source.write_text("FIXED_LAMBDA = 0.25\n")
            with patch(f"{MODULE}.train") as training:
                with self.assertRaises(ValueError):
                    experiment.train()
                with self.assertRaises(ValueError):
                    experiment.evaluate()
                training.assert_not_called()

    def test_production_preflight_profiles_only_exact_fixed_config(self):
        """Forward the fixed method and complete caller config without scheduling ERM or CUE."""
        config = self.production_config()
        profile = self.production_profile(config)
        with self.experiment(config=config, expected_initial_weights="a" * 64) as experiment:
            with (
                patch(f"{MODULE}.seed_everything"),
                patch(f"{MODULE}.profile_dataset", return_value=profile) as measured,
                patch(f"{MODULE}.verify_profile") as verified,
                patch(f"{MODULE}.estimated_run_seconds", return_value=100),
            ):
                self.assertEqual(experiment.preflight(), profile)
            measured.assert_called_once_with(
                ROOT,
                experiment.manifest,
                config,
                self.directory / "preflight.json",
                steps=10,
                methods=("fixed_global_local",),
            )
            verified.assert_called_once_with(
                profile, ROOT, experiment.manifest, config, methods=("fixed_global_local",)
            )
            self.assertTrue(experiment.ready)

    def test_cached_preflight_rejects_changed_coefficient_and_step_count(self):
        """Reject scalar recipe changes that the memory-only preflight signature cannot detect."""
        config = self.production_config()
        for name in ("coefficient", "steps"):
            with self.subTest(name=name):
                profile = copy.deepcopy(self.production_profile(config))
                if name == "coefficient":
                    profile["active"]["config"]["fixed_lambda"] = 0.25
                else:
                    profile["optimizer_steps_per_method"] = 11
                directory = self.folder / name
                with self.experiment(config=config, directory=directory) as experiment:
                    atomic_json(directory / "preflight.json", profile)
                    with (
                        patch(f"{MODULE}.seed_everything"),
                        patch(f"{MODULE}.verify_profile"),
                        patch(f"{MODULE}.profile_dataset") as measured,
                        patch(f"{MODULE}.train") as training,
                    ):
                        with self.assertRaises(ValueError):
                            experiment.preflight()
                        with self.assertRaises(RuntimeError):
                            experiment.train()
                        measured.assert_not_called()
                        training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
