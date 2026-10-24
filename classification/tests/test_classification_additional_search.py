"""Verify additional CUE trials, strict validation-reference selection, and deferred final testing."""

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
from clvision.engine import make_model
from clvision.experiment import Comparison
from clvision.experiment_runtime import preflight_search
from clvision.experiment_settings import Search, validate_stages

from cue_routing.io import atomic_json, file_hash, read_json, seed_everything

ROOT = Path(__file__).resolve().parents[2]


class AdditionalSearchTests(unittest.TestCase):
    """Run real small CPU trials; synthetic reference scores exercise both final-test branches."""

    def setUp(self):
        """Keep all generated inputs outside executable source and serialize training on two CPU threads."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.data_root = self.folder / "data"
        self.manifest = data.make_fixture(self.data_root)
        self.directory = self.folder / "study"
        self.config = Config(
            method="cue",
            activation="preact",
            initialization="msra",
            normalization="zipfls",
            gate_tau=0.3,
            gate_temperature=0.05,
            joint_floor=0.5,
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
        )
        self.search = Search([{"gate_temperature": 0.025}, {"gate_temperature": 0.1}])
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(display_options(DisplayOptions(progress="off")))
        seed_everything(self.config.seed, self.config.device)
        self.initial = make_model(self.config).shared_state_hash()

    def reference(self, *, accuracy=1.0, nll=1e6, config=None):
        """Provide an explicitly synthetic incumbent with the same split and shared initialization."""
        return {
            "config": (config or self.config).to_dict(),
            "validation": {"accuracy": accuracy, "nll": nll, "n": 4},
            "dataset_id": self.manifest["dataset_id"],
            "initial_shared_model_sha256": self.initial,
            "provenance": {"kind": "synthetic test reference"},
        }

    def experiment(self, **overrides):
        """Use only public caller inputs, with no baseline enabled."""
        return Comparison(
            **{
                "root": ROOT,
                "data_root": self.data_root,
                "directory": self.directory,
                "config": self.config,
                "search": self.search,
                "include_baseline": False,
                **overrides,
            }
        )

    def test_new_winner_only_two_cue_runs_one_test_and_completed_resume(self):
        """Require closed selection before test reads and preserve one test event after resuming."""
        from clvision.experiment import train as actual_train

        order, opened = [], []
        original = data.read_array
        search = Search(self.search.candidates, metric="nll")

        def record(config, *args, **kwargs):
            """Observe the actual training calls without replacing SGD or validation."""
            order.append((config.method, kwargs.get("phase")))
            return actual_train(config, *args, **kwargs)

        def guard(root, descriptor):
            """Require every new candidate completed and the selected copy sealed before test reads."""
            if "_test_" in descriptor["path"]:
                self.assertEqual(
                    len(list((self.directory / "development").rglob("completion.json"))), 2
                )
                self.assertTrue((self.directory / "selection.json").exists())
                self.assertTrue((self.directory / "study_seal.json").exists())
                self.assertFalse(list((self.directory / "development").rglob("test_access.json")))
                opened.append(descriptor["path"])
            return original(root, descriptor)

        with (
            patch("clvision.experiment.train", side_effect=record),
            patch("clvision.data.read_array", side_effect=guard),
        ):
            with self.experiment(reference=self.reference(), search=search) as run:
                result = run.run()
        self.assertEqual(order, [("cue", "development")] * 2)
        self.assertTrue(opened)
        self.assertFalse((self.directory / "main/seed0_erm").exists())
        self.assertFalse(result["selection"]["reference_retained"])
        self.assertEqual(set(result["test_metrics"]), {"cue"})
        self.assertEqual(result["test_metrics"]["cue"]["n"], 12)
        self.assertEqual(result["protocol"]["maximum_training_runs"], 2)
        paths = list(self.directory.rglob("completion.json")) + list(
            self.directory.rglob("test_predictions.pt")
        )
        before = {str(path): file_hash(path) for path in paths}
        with self.experiment(reference=self.reference(), search=search, resume=True) as run:
            again = run.run()
        self.assertEqual(result["test_metrics"], again["test_metrics"])
        self.assertEqual(before, {str(path): file_hash(path) for path in paths})
        access = list(self.directory.rglob("test_access.json"))
        self.assertEqual(len(access), 1)
        self.assertEqual(len(read_json(access[0])), 1)

    def test_retained_reference_never_opens_test_or_promotes_checkpoint(self):
        """A reference at the maximum possible accuracy cannot be strictly beaten on validation."""
        original = data.read_array

        def guard(root, descriptor):
            """Deny every held-out array read, including attempted verification before final selection."""
            self.assertNotIn("_test_", descriptor["path"])
            return original(root, descriptor)

        with patch("clvision.data.read_array", side_effect=guard):
            with self.experiment(reference=self.reference()) as run:
                result = run.run()
            with self.experiment(reference=self.reference(), resume=True) as run:
                again = run.run()
        self.assertTrue(result["selection"]["reference_retained"])
        self.assertIsNone(result["selection"]["selected"])
        self.assertFalse(result["final_test_accessed"])
        self.assertEqual(result["test_metrics"], {})
        self.assertEqual(result["selection"], again["selection"])
        self.assertFalse((self.directory / "main").exists())
        self.assertFalse(list(self.directory.rglob("test_access.json")))
        self.assertEqual(len(list(self.directory.rglob("completion.json"))), 2)

    def test_accuracy_tie_keeps_reference_even_with_better_candidate_nll(self):
        """Only a strict primary-metric improvement may unlock the selected candidate's test."""
        for score, retained in ((0.75, True), (0.7502, False)):
            with (
                self.subTest(score=score),
                self.experiment(
                    directory=self.folder / str(score), reference=self.reference(accuracy=0.75)
                ) as run,
            ):
                run.journal["decisions"] = [{"index": i, "action": "run"} for i in range(2)]
                rows = []
                for i, config in enumerate(run.candidates):
                    rows.append(
                        {
                            "name": f"cue{i}",
                            "index": i,
                            "config": config.to_dict(),
                            "validation": {"accuracy": score - i * 0.01, "nll": 1.0, "n": 4},
                        }
                    )
                    atomic_json(
                        run.directory / "development" / f"cue{i}" / "contract.json",
                        {
                            "initial_shared_model_sha256": self.initial,
                        },
                    )
                with (
                    patch("clvision.experiment.collect_candidates", return_value=rows),
                    patch("clvision.experiment.promote_winner") as promote,
                ):
                    decision = run._selection()
                self.assertEqual(decision["reference_retained"], retained)
                self.assertEqual(promote.call_count, int(not retained))

    def test_reference_config_split_and_initial_weights_must_match(self):
        """Reject incompatible historical evidence before expensive training or final testing."""
        with self.assertRaisesRegex(ValueError, "common settings"):
            self.experiment(config=replace(self.config, seed=1), reference=self.reference())
        reference = self.reference()
        reference["dataset_id"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Reference dataset"):
            with self.experiment(reference=reference):
                pass
        config = replace(self.config, device="cuda", debug=False, debug_max_batches=0)
        with self.experiment(config=config, reference=self.reference(config=config)) as run:
            bad = {"methods": {"cue": {"initial_shared_model_sha256": "b" * 64}}}
            with (
                patch("clvision.experiment.seed_everything"),
                patch("clvision.experiment.preflight_search", return_value=(None, [bad, bad])),
                patch("clvision.experiment.train") as training,
            ):
                with self.assertRaisesRegex(ValueError, "initial weights"):
                    run.preflight()
                with self.assertRaisesRegex(RuntimeError, "preflight"):
                    run.tune()
                training.assert_not_called()

    def test_cue_only_preflight_never_profiles_erm_and_failed_preflight_never_trains(self):
        """Exercise candidate-specific profiler inputs while avoiding CUDA in a CPU test suite."""
        config = replace(self.config, device="cuda", debug=False, debug_max_batches=0)
        candidates = self.search.resolve(replace(config, method="erm"))
        calls = []

        def profile(root, manifest, candidate, destination, *, steps, methods):
            """Supply profiler identity fields to isolate scheduling from device availability."""
            calls.append((candidate, methods))
            return {
                "active": {"config": candidate.to_dict()},
                "optimizer_steps_per_method": steps,
                "methods": {
                    "cue": {
                        "initial_shared_model_sha256": self.initial,
                        "first_augmented_batch_sha256": "same",
                    }
                },
            }

        with (
            patch("clvision.experiment_runtime.profile_dataset", side_effect=profile),
            patch("clvision.experiment_runtime.verify_profile"),
        ):
            baseline, profiles = preflight_search(
                ROOT, self.directory, self.manifest, config, candidates, include_baseline=False
            )
        self.assertIsNone(baseline)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(calls, [(candidate, ("cue",)) for candidate in candidates])
        with self.experiment() as run:
            with (
                patch(
                    "clvision.experiment.preflight_search",
                    side_effect=RuntimeError("preflight failed"),
                ),
                patch("clvision.experiment.train") as training,
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    run.preflight()
                with self.assertRaisesRegex(RuntimeError, "preflight"):
                    run.tune()
                training.assert_not_called()
            run.ready = True
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                run.baseline()
            with self.assertRaisesRegex(RuntimeError, "test remains sealed"):
                run.evaluate()
        with self.assertRaises(ValueError):
            validate_stages(("preflight", "tune", "baseline", "evaluate"), include_baseline=False)

    def test_reference_mutation_on_resume_cannot_change_selection_threshold(self):
        """Freeze previous validation evidence along with candidate settings before training."""
        reference = self.reference()
        with self.experiment(reference=reference):
            pass
        changed = copy.deepcopy(reference)
        changed["validation"]["accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "scope or source changed"):
            with self.experiment(reference=changed, resume=True):
                pass


if __name__ == "__main__":
    unittest.main()
