"""Check public experiment inputs, configuration precedence, and fail-fast driver forwarding."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clvision.arguments import add_study_arguments, load_study_arguments, study_preview
from clvision.cli import main

ROOT = Path(__file__).resolve().parents[2]


class ClassificationArgumentTests(unittest.TestCase):
    """Exercise real argument parsing without accessing datasets or starting training."""

    def parse(self, tokens):
        """Use the same argument definitions as both supported entry points."""
        parser = argparse.ArgumentParser()
        add_study_arguments(parser, ROOT)
        return parser.parse_args(tokens)

    def test_default_and_custom_seed_counts(self):
        """Count all development, paired main, and mechanism jobs from explicit inputs."""
        defaults = study_preview(load_study_arguments(self.parse([])))
        self.assertEqual(defaults["training_runs"]["total"], 50)
        settings = load_study_arguments(
            self.parse(["--seeds", "7", "11", "19", "--mechanism-seeds", "23"])
        )
        preview = study_preview(settings)
        self.assertEqual(preview["settings"]["main_seeds"], [7, 11, 19])
        self.assertEqual(preview["settings"]["mechanism_seeds"], [23])
        self.assertEqual(preview["training_runs"]["total"], 54)

    def test_cli_overrides_dataset_and_schedule_settings(self):
        """An explicit flag takes precedence even when a JSON dataset has a specific value."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recipe.json"
            path.write_text(
                json.dumps(
                    {
                        "training": {"learning_rate": 0.02},
                        "natural_schedule": {"lr_decay": 0.3},
                        "dataset_overrides": {"cifar100": {"learning_rate": 0.4, "epochs": 200}},
                    }
                )
            )
            settings = load_study_arguments(
                self.parse(
                    [
                        "--config",
                        str(path),
                        "--learning-rate",
                        "0.03",
                        "--lr-decay",
                        "0.2",
                        "--epochs",
                        "20",
                        "--milestones",
                        "10",
                        "15",
                        "--batch-size",
                        "64",
                        "--mechanism-epochs",
                        "10",
                        "--mechanism-milestones",
                        "5",
                        "8",
                    ]
                )
            )
            preview = study_preview(settings)
            for dataset, config in preview["resolved_recipes"].items():
                self.assertEqual(config["learning_rate"], 0.03)
                self.assertEqual(config["lr_decay"], 0.2)
                self.assertEqual(config["batch_size"], 64)
                self.assertEqual(config["epochs"], 10 if dataset == "dominoes" else 20)

    def test_invalid_settings_fail_before_dataset_access(self):
        """Reject malformed seeds and incompatible schedules before any expensive work."""
        for tokens in (["--seeds", "1", "1"], ["--epochs", "10"], ["--batch-size", "1"]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                load_study_arguments(self.parse(tokens))
        settings = load_study_arguments(self.parse(["--epochs", "10", "--milestones"]))
        self.assertEqual(settings.recipe("cifar100").milestones, ())

    def test_plan_never_reads_data_or_starts_study(self):
        """A real CLI preview is safe before data preparation and CUDA availability."""
        output = io.StringIO()
        with (
            patch("sys.argv", ["experiment.py", "study", "--seeds", "0", "1", "2", "--dry-run"]),
            patch("clvision.cli.load_manifest", side_effect=AssertionError("must not open data")),
            patch("clvision.cli.run_study", side_effect=AssertionError("must not train")),
            contextlib.redirect_stdout(output),
        ):
            main(ROOT)
        result = json.loads(output.getvalue())
        self.assertEqual(result["training_runs"]["total"], 66)
        self.assertFalse(result["training_performed"])

    def test_malformed_nested_configuration_has_actionable_error(self):
        """Reject invalid object shapes before merging explicit CLI overrides."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recipe.json"
            for value in (
                {"natural_schedule": []},
                {"dataset_overrides": []},
                {"dataset_overrides": {"cifar100": []}},
                {"training": None},
            ):
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "JSON object"):
                    load_study_arguments(self.parse(["--config", str(path)]))

    def test_driver_forwards_explicit_inputs_and_unbuffered_python(self):
        """The convenient driver preserves custom flags, paths with spaces, and failure handling."""
        specification = importlib.util.spec_from_file_location(
            "study_driver", ROOT / "classification/run.py"
        )
        driver = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(driver)
        with (
            patch(
                "sys.argv",
                [
                    "run.py",
                    "train",
                    "--seeds",
                    "8",
                    "9",
                    "--learning-rate",
                    "0.02",
                    "--runs-dir",
                    "/tmp/study with spaces",
                    "--workers",
                    "0",
                ],
            ),
            patch.object(driver.subprocess, "run") as call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            driver.main()
        command = call.call_args.args[0]
        self.assertEqual(command[1], "-u")
        self.assertIn(str(Path("/tmp/study with spaces").resolve()), command)
        self.assertEqual(
            command[command.index("--seeds") + 1 : command.index("--seeds") + 3], ["8", "9"]
        )
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.02")
        self.assertTrue(call.call_args.kwargs["check"])
