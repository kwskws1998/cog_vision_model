"""Progress output, accepted-step semantics, telemetry, and bitwise resume regressions."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.arguments import add_study_arguments, study_argument_tokens
from clvision.cue_settings import add_cue_arguments, load_cue_arguments
from clvision.data import make_fixture
from clvision.display import (
    OPTIONS,
    BatchProgress,
    DisplayOptions,
    TrainingLog,
    display_options,
    options_from_arguments,
)
from clvision.engine import evaluate_loader, optimize_batch, train
from test_classification_training import debug_config, equal_states

ROOT = Path(__file__).resolve().parents[2]


class DisplayTests(unittest.TestCase):
    """Use real CPU optimization and controlled clocks rather than treating text as a benchmark."""

    def setUp(self):
        """Allocate isolated training and logging artifacts."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        """Remove only this test's temporary artifacts."""
        self.temporary.cleanup()

    def rows(self, folder):
        """Read complete JSON records from one append-only run journal."""
        return [
            json.loads(line) for line in (folder / "training_log.jsonl").read_text().splitlines()
        ]

    def test_modes_support_tee_plain_logs_and_disabled_bars(self):
        """A non-TTY pipe can show live bars while plain mode remains free of terminal controls."""
        for mode in ("bar", "plain", "off"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), display_options(DisplayOptions(progress=mode)):
                with BatchProgress("Train fixture", 3) as meter:
                    meter.update(1, loss="1.2")
                    meter.update(3, loss="0.8")
            text = output.getvalue()
            if mode == "bar":
                self.assertIn("\r", text)
                self.assertIn("3/3", text)
                self.assertIn("100%", text)
            elif mode == "plain":
                self.assertIn("100.0% 3/3", text)
                self.assertNotIn("\r", text)
                self.assertNotIn("\x1b", text)
            else:
                self.assertEqual(text, "")

    def test_resumed_rate_uses_only_new_steps(self):
        """A saved counter must not make the first resumed seconds appear hundreds of times faster."""
        clock = {"now": 0.0}
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            display_options(DisplayOptions(progress="plain")),
            patch("clvision.display.time.monotonic", side_effect=lambda: clock["now"]),
        ):
            with BatchProgress("Train resumed", 1000, initial=600) as meter:
                clock["now"] = 5.0
                meter.update(610)
        text = output.getvalue()
        self.assertIn("60.0% 600/1000", text)
        self.assertIn("steps/s=2.00", text)
        self.assertIn("step_remaining~=03:15", text)
        self.assertIn("status=paused", text)
        self.assertNotIn("100.0%", text)

    def test_failure_closes_bar_without_inventing_completed_work(self):
        """Unexpected failures and keyboard interruption retain the last accepted step."""
        for exception, status in (
            (RuntimeError("bad batch"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            output = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                display_options(DisplayOptions(progress="bar")),
            ):
                with self.assertRaises(type(exception)):
                    with BatchProgress("Train interrupted", 10) as meter:
                        meter.update(2)
                        raise exception
                self.assertTrue(meter.bar.disable)
            text = output.getvalue()
            self.assertIn("2/10", text)
            self.assertIn(f"status={status}", text)
            self.assertNotIn("100%", text)

    def test_weighted_windows_and_short_tail_are_logged_once(self):
        """Uneven batches contribute by sample count and the final short window is not discarded."""
        config = debug_config()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            display_options(DisplayOptions(progress="off", logging_steps=2)),
        ):
            with TrainingLog(config, self.folder, 3, 0) as logging:
                logging.step(1, 1, 1, 3, 2, {"loss": 2.0, "train_accuracy": 0.5}, 0.1)
                logging.step(2, 1, 2, 3, 1, {"loss": 8.0, "train_accuracy": 1.0}, 0.1)
                logging.step(3, 1, 3, 3, 1, {"loss": 1.0, "train_accuracy": 0.0}, 0.1)
        rows = self.rows(self.folder)
        logs = [row for row in rows if row["event"] == "train"]
        self.assertEqual([row["step"] for row in logs], [2, 3])
        self.assertEqual(logs[0]["loss"], 4.0)
        self.assertAlmostEqual(logs[0]["train_accuracy"], 2 / 3)
        self.assertEqual((logs[0]["examples"], logs[0]["window_steps"]), (3, 2))
        self.assertEqual(logs[1]["loss"], 1.0)
        self.assertEqual(rows[-1]["status"], "complete")

    def test_cli_logging_options_are_validated_forwarded_and_not_hyperparameters(self):
        """Presentation flags have one meaning across entrypoints and do not change recipes."""
        parser = argparse.ArgumentParser()
        add_cue_arguments(parser, ROOT)
        basic = parser.parse_args([])
        changed = parser.parse_args(
            ["--progress", "plain", "--logging-steps", "7", "--progress-interval", "3"]
        )
        self.assertEqual(load_cue_arguments(basic), load_cue_arguments(changed))
        self.assertEqual(options_from_arguments(changed), DisplayOptions("plain", 7, 3))
        parser = argparse.ArgumentParser()
        add_study_arguments(parser, ROOT)
        args = parser.parse_args(["--progress", "off", "--logging-steps", "13"])
        forwarded = parser.parse_args(study_argument_tokens(args))
        self.assertEqual(options_from_arguments(forwarded), DisplayOptions("off", 13, 1))
        for options in (
            DisplayOptions(logging_steps=0),
            DisplayOptions(logging_steps=True),
            DisplayOptions(progress_interval=float("nan")),
            DisplayOptions(progress_interval=0),
            DisplayOptions(progress="bad"),
        ):
            with self.assertRaises(ValueError):
                options.validate()
        original = OPTIONS.get()
        with self.assertRaises(RuntimeError):
            with display_options(DisplayOptions(progress="off")):
                raise RuntimeError("exit")
        self.assertEqual(OPTIONS.get(), original)

    def test_resume_across_display_modes_preserves_model_optimizer_and_rng(self):
        """Live bars, summaries, and resumed logging must not modify stochastic training semantics."""
        manifest = make_fixture(self.folder / "data")
        config = replace(debug_config(), epochs=2, milestones=(1,))
        full, resumed = self.folder / "full", self.folder / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            with display_options(DisplayOptions(progress="off", logging_steps=200)):
                train(config, manifest, full, ROOT, phase="debug")
            with display_options(DisplayOptions(progress="bar", logging_steps=1)):
                train(config, manifest, resumed, ROOT, phase="debug", stop_after=1)
            with display_options(DisplayOptions(progress="plain", logging_steps=2)):
                train(config, manifest, resumed, ROOT, phase="debug", resume=True)
        first, second = [
            torch.load(folder / "last.pt", weights_only=True) for folder in (full, resumed)
        ]
        for key in ("model", "optimizer", "scaler", "steps", "rng"):
            self.assertTrue(equal_states(first[key], second[key]), key)
        rows = self.rows(resumed)
        starts = [row for row in rows if row["event"] == "session_start"]
        self.assertEqual([row["resumed_from_step"] for row in starts], [0, first["steps"] // 2])
        self.assertNotEqual(starts[0]["session_id"], starts[1]["session_id"])
        self.assertEqual(
            [row["status"] for row in rows if row["event"] == "session_end"], ["paused", "complete"]
        )
        epochs = [row for row in rows if row["event"] == "epoch"]
        self.assertEqual([row["epoch"] for row in epochs], [1, 2])
        self.assertIn("checkpoint_seconds", epochs[-1])

    def test_rejected_optimizer_update_does_not_advance_training_progress(self):
        """A real accepted CPU update followed by a failure logs exactly one accepted step."""
        manifest = make_fixture(self.folder / "data")
        config = replace(debug_config(), batch_size=4, debug_max_batches=0)
        calls = {"n": 0}

        def fail_second(*args):
            """Raise before the second optimizer update without mocking the first update."""
            calls["n"] += 1
            if calls["n"] == 2:
                raise FloatingPointError("rejected update")
            return optimize_batch(*args)

        directory = self.folder / "failed"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            display_options(DisplayOptions(progress="plain")),
            patch("clvision.engine.optimize_batch", side_effect=fail_second),
        ):
            with self.assertRaisesRegex(FloatingPointError, "rejected update"):
                train(config, manifest, directory, ROOT, phase="debug")
        rows = self.rows(directory)
        self.assertEqual((rows[-1]["step"], rows[-1]["status"]), (1, "failed"))
        self.assertFalse((directory / "completion.json").exists())

    def test_failed_evaluation_does_not_complete_its_progress_bar(self):
        """Invalid predictions are rejected before incrementing the evaluation batch counter."""

        class Model(torch.nn.Module):
            """Return finite predictions for one batch, then invalid predictions."""

            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                return torch.full((len(images), 100), 0.0 if self.calls == 1 else float("nan"))

        output = io.StringIO()
        loader = [(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))] * 2
        with contextlib.redirect_stdout(output), display_options(DisplayOptions(progress="plain")):
            with self.assertRaises(FloatingPointError):
                evaluate_loader(Model(), loader, debug_config(), progress_name="Eval fixture")
        self.assertIn("50.0% 1/2", output.getvalue())
        self.assertIn("status=failed", output.getvalue())
        self.assertNotIn("100.0%", output.getvalue())
