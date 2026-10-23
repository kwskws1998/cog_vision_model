"""Regression checks for whole-study ETA scope, observed timing, resume, and failure reporting."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from clvision.config import Config
from clvision.progress import CURRENT_PROGRESS, StudyProgress, duration, progress_event

from cue_routing.io import atomic_json, read_json


class StudyProgressTests(unittest.TestCase):
    """Use known image counts and rates to verify ETA calculations without a GPU."""

    def setUp(self):
        """Define a ten-epoch candidate plus a ten-epoch main run with corruption testing."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        config = Config(method="erm", epochs=10, milestones=(5,)).to_dict()
        self.plan = {
            "development_trials": {"cifar100": [{"name": "erm_candidate0", "config": config}]},
            "resolved_recipes": {"cifar100": config},
            "main_methods": ["erm"],
            "main_seeds": [3],
            "mechanism_methods": [],
            "mechanism_seeds": [],
        }
        self.manifests = {
            "cifar100": {
                "splits": {
                    "train": list(range(100)),
                    "validation": list(range(20)),
                    "test": list(range(10)),
                }
            }
        }
        self.profiles = {
            "cifar100": {
                "methods": {
                    "erm": {
                        "training_images_per_second": 10.0,
                        "evaluation_images_per_second": 5.0,
                    }
                }
            }
        }
        self.main = self.folder / "main/cifar100/seed3_erm"
        self.candidate = self.folder / "development/cifar100/erm_candidate0"

    def tearDown(self):
        """Remove only isolated telemetry fixtures."""
        self.temporary.cleanup()

    def tracker(self, profiles=True):
        """Construct a new tracker with optional initial hardware measurements."""
        return StudyProgress(
            self.folder, self.plan, self.manifests, self.profiles if profiles else {}
        )

    def test_initial_estimate_includes_development_validation_and_75_corruptions(self):
        """The complete estimate includes all declared phases rather than only the active run."""
        tracker = self.tracker()
        training, evaluation, unknown = tracker.estimate()
        self.assertEqual(training, 280)
        self.assertEqual(evaluation, 152)
        self.assertEqual(unknown, [])
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.emit(force=True)
        report = read_json(self.folder / "progress.json")
        self.assertEqual(report["estimated_remaining_seconds"], 432)
        self.assertEqual(report["total_training_runs"], 2)
        self.assertEqual(report["total_corruption_cells"], 75)
        self.assertTrue(report["estimated_finish_kst"].endswith("+09:00"))

    def test_observed_epochs_replace_provisional_rates_and_keep_pending_validation(self):
        """Epoch measurements include writes; partial training never discards pending validation."""
        tracker = self.tracker()
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.update(
                self.main,
                "train_epoch",
                epoch=1,
                epoch_seconds=20,
                training_seconds=12,
                validation_seconds=4,
            )
            self.assertEqual(tracker.estimate()[0], 380)
            tracker.update(self.main, "train_batch", epoch=2, batch=1, batches=2)
            self.assertEqual(tracker.estimate()[0], 374)
            tracker.update(self.main, "train_batch", epoch=2, batch=2, batches=2)
            self.assertEqual(tracker.estimate()[0], 368)

    def test_resume_removes_finished_epochs_runs_and_corruption_cells(self):
        """A restart estimates remaining work rather than charging completed work again."""
        atomic_json(self.candidate / "completion.json", {})
        atomic_json(self.main / "history.json", [{"epoch": 3, "elapsed_seconds": 60}])
        atomic_json(self.main / "test_metrics.json", {})
        atomic_json(self.main / "corruption_cells.json", {"cells": {"a/1": {}, "a/2": {}}})
        tracker = self.tracker()
        self.assertEqual(tracker.estimate(), (140, 146, []))
        self.assertEqual(sum(job["complete"] for job in tracker.jobs.values()), 1)

    def test_unmeasured_eta_is_unknown_and_failure_is_recorded(self):
        """Missing throughput must not produce a falsely precise time or a successful status."""
        tracker = self.tracker(profiles=False)
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.emit(force=True)
            self.assertIsNone(
                read_json(self.folder / "progress.json")["estimated_remaining_seconds"]
            )
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with tracker:
                    raise RuntimeError("training failed")
        report = read_json(self.folder / "progress.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "training failed")
        self.assertIsNone(report["estimated_finish_kst"])
        self.assertIsNone(CURRENT_PROGRESS.get())

    def test_evaluation_progress_reaches_zero_only_after_clean_and_corruptions(self):
        """Finishing training keeps the final evaluation budget visible until evaluation completes."""
        tracker = self.tracker()
        with contextlib.redirect_stdout(io.StringIO()), tracker:
            progress_event(self.candidate, "train_complete")
            progress_event(self.main, "train_complete")
            self.assertEqual(tracker.estimate(), (0, 152, []))
            progress_event(self.main, "clean_complete", seconds=2)
            progress_event(self.main, "corruption_cell", completed_cells=75, seconds=2)
            self.assertEqual(tracker.estimate(), (0, 0, []))
        report = read_json(self.folder / "progress.json")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_training_runs"], 2)
        self.assertEqual(report["completed_corruption_cells"], 75)
        self.assertIsNone(CURRENT_PROGRESS.get())

    def test_dominoes_estimate_counts_counterfactual_pairs_and_three_panel_passes(self):
        """Mechanism clean evaluation costs six predictions per held-out vehicle."""
        self.plan["development_trials"] = {}
        self.plan["resolved_recipes"] = {"dominoes": Config(dataset="dominoes").to_dict()}
        self.plan["mechanism_methods"] = ["erm"]
        self.plan["mechanism_seeds"] = [0]
        self.manifests["dominoes"] = self.manifests["cifar100"]
        tracker = self.tracker()
        job = next(iter(tracker.jobs.values()))
        self.assertEqual(job["clean_images"], 60)
        self.assertEqual(job["corruption_cells"], 0)

    def test_duration_handles_unknown_and_multiday_values(self):
        """Human-readable times retain hours beyond one day and distinguish unknown time."""
        self.assertEqual(duration(None), "estimating")
        self.assertEqual(duration(90061), "25h01m01s")
