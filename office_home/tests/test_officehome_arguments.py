"""Office-Home workflow forwarding and fail-fast CLI configuration regressions."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohvision.cli import main

ROOT = Path(__file__).resolve().parents[2]


class OfficeHomeArgumentsTests(unittest.TestCase):
    """Keep profiling, source smoke, and matrix recipes aligned under explicit overrides."""

    def driver(self):
        """Load the standalone workflow without invoking its main entrypoint."""
        specification = importlib.util.spec_from_file_location(
            "officehome_argument_driver", ROOT / "office_home/run.py"
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def test_training_flags_reach_profile_smoke_and_matrix(self):
        driver = self.driver()
        arguments = driver.build_parser().parse_args(
            [
                "train",
                "--steps",
                "7000",
                "--batch-per-domain",
                "8",
                "--learning-rate",
                "0.0002",
                "--no-freeze-bn",
                "--precision",
                "bf16",
                "--seeds",
                "42",
                "43",
                "--smoke-steps",
                "3",
            ]
        )
        config, commands = driver.build_commands(arguments)
        self.assertEqual(config.steps, 7000)
        calls = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ohvision.cli.profile_memory",
                return_value={"measured": True, "passes_budget": True},
            ) as profile,
            patch("ohvision.cli.real_data_smoke", return_value={}) as smoke,
            patch("ohvision.cli.matrix", return_value={}) as matrix,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for index, command in enumerate(commands):
                if index == 0:
                    command = [*command, "--output", str(Path(temporary) / "profile.json")]
                main(ROOT, command)
            calls = [profile.call_args.args[0], smoke.call_args.args[2], matrix.call_args.args[0]]
        for current in calls:
            self.assertEqual(current.steps, 7000)
            self.assertEqual(current.batch_per_domain, 8)
            self.assertEqual(current.learning_rate, 0.0002)
            self.assertEqual(current.precision, "bf16")
            self.assertFalse(current.freeze_bn)
        self.assertEqual(smoke.call_args.args[3], 3)
        self.assertEqual(matrix.call_args.args[4], [42, 43])
        self.assertTrue(matrix.call_args.args[5])

    def test_prepare_preserves_explicit_smoke_settings(self):
        driver = self.driver()
        arguments = driver.build_parser().parse_args(
            ["prepare", "--batch-per-domain", "2", "--smoke-steps", "4"]
        )
        config, commands = driver.build_commands(arguments)
        with (
            patch("ohvision.cli.real_data_smoke", return_value={}) as smoke,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(ROOT, commands[-1])
        self.assertEqual(smoke.call_args.args[2], config)
        self.assertEqual((config.batch_per_domain, config.precision), (2, "fp32"))
        self.assertEqual(smoke.call_args.args[3], 4)

    def test_dry_run_validates_without_data_or_execution(self):
        driver = self.driver()
        output = io.StringIO()
        with (
            patch.object(driver.subprocess, "run") as execution,
            contextlib.redirect_stdout(output),
        ):
            driver.main(
                [
                    "train",
                    "--dry-run",
                    "--manifest",
                    "/nonexistent/manifest.json",
                    "--seeds",
                    "7",
                    "--no-evaluate",
                ]
            )
        execution.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan["commands"]), 3)
        self.assertNotIn("--evaluate", plan["commands"][-1])
        self.assertEqual(plan["commands"][-1][-2:], ["--seeds", "7"])

    def test_invalid_workflow_settings_fail_before_execution(self):
        driver = self.driver()
        for options in (
            ["--seeds", "42", "42"],
            ["--steps", "2"],
            ["--smoke-steps", "0"],
            ["--device", "cpu", "--precision", "fp32"],
            ["--debug"],
        ):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(driver.subprocess, "run") as execution,
                self.assertRaises(SystemExit),
            ):
                driver.main(["train", "--dry-run", *options])
            execution.assert_not_called()

    def test_officehome_specific_field_and_target_reach_training(self):
        with (
            patch("ohvision.cli.train", return_value={}) as training,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                ROOT,
                [
                    "train",
                    "--run-dir",
                    "unused",
                    "--target",
                    "Clipart",
                    "--seed",
                    "42",
                    "--data-variant",
                    "resized_256_square",
                    "--learning-rate",
                    "0.0001",
                ],
            )
        config = training.call_args.args[0]
        self.assertEqual(
            (config.target, config.seed, config.data_variant), ("Clipart", 42, "resized_256_square")
        )
        self.assertEqual(config.learning_rate, 0.0001)


if __name__ == "__main__":
    unittest.main()
