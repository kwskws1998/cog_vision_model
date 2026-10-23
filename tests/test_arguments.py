"""Configuration precedence, explicit overrides, and smoke command boundary regressions."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from cue_routing.arguments import (
    add_config_arguments,
    config_argument_tokens,
    load_config_arguments,
    load_data_smoke_config,
)
from cue_routing.cli import main
from cue_routing.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationArgumentsTests(unittest.TestCase):
    """Verify explicit CLI settings preserve JSON values and use final merged validation."""

    def parser(self):
        """Create a standalone configuration parser using the production shared helper."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path)
        add_config_arguments(parser, ExperimentConfig)
        return parser

    def test_omitted_flags_preserve_json_and_explicit_false_and_zero_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"learning_rate": 0.002, "workers": 3, "freeze_bn": True}))
            args = self.parser().parse_args(
                ["--config", str(path), "--no-freeze-bn", "--workers", "0"]
            )
            config = load_config_arguments(args, ExperimentConfig)
            self.assertEqual(config.learning_rate, 0.002)
            self.assertEqual(config.workers, 0)
            self.assertFalse(config.freeze_bn)
            self.assertEqual(
                config_argument_tokens(args, ExperimentConfig), ["--workers", "0", "--no-freeze-bn"]
            )

    def test_final_combination_is_validated_after_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"device": "cpu", "precision": "fp16", "steps": 2}))
            args = self.parser().parse_args(
                ["--config", str(path), "--precision", "fp32", "--debug"]
            )
            config = load_config_arguments(args, ExperimentConfig)
            self.assertEqual(config.precision, "fp32")
            self.assertEqual(config.steps, 2)
            with self.assertRaisesRegex(ValueError, "CPU verification"):
                load_config_arguments(
                    self.parser().parse_args(["--config", str(path)]), ExperimentConfig
                )

    def test_every_field_can_be_forwarded_without_changing_types(self):
        config = ExperimentConfig()
        namespace = argparse.Namespace(**config.to_dict())
        tokens = config_argument_tokens(namespace, ExperimentConfig)
        parsed = self.parser().parse_args(tokens)
        self.assertEqual(load_config_arguments(parsed, ExperimentConfig), config)
        self.assertEqual({field.name for field in fields(config)}, set(vars(parsed)) - {"config"})

    def test_unknown_json_keys_and_invalid_numeric_settings_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"learnig_rate": 0.1}')
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_config_arguments(
                    self.parser().parse_args(["--config", str(path)]), ExperimentConfig
                )
        for option in (["--learning-rate", "nan"], ["--steps", "2"], ["--no-pretrained"]):
            with self.subTest(option=option), self.assertRaises(ValueError):
                load_config_arguments(self.parser().parse_args(option), ExperimentConfig)

    def test_cpu_smoke_defaults_precede_user_overrides(self):
        args = self.parser().parse_args(["--batch-per-domain", "4", "--workers", "2"])
        config, steps = load_data_smoke_config(args, ExperimentConfig)
        self.assertEqual((config.device, config.precision, config.steps), ("cpu", "fp32", 1))
        self.assertEqual((config.batch_per_domain, config.workers, steps), (4, 2, 1))
        invalid = self.parser().parse_args(["--precision", "bf16"])
        with self.assertRaisesRegex(ValueError, "CPU verification"):
            load_data_smoke_config(invalid, ExperimentConfig)
        for flags in (["--no-pretrained"], ["--image-size", "64"]):
            with (
                self.subTest(flags=flags),
                self.assertRaisesRegex(ValueError, "pretrained weights"),
            ):
                load_data_smoke_config(self.parser().parse_args(flags), ExperimentConfig)

    def test_data_smoke_steps_do_not_replace_training_budget(self):
        with (
            patch("cue_routing.cli.real_data_smoke", return_value={}) as smoke,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                ROOT,
                [
                    "data-smoke",
                    "--device",
                    "cuda",
                    "--steps",
                    "7",
                    "--training-steps",
                    "6000",
                    "--batch-per-domain",
                    "8",
                ],
            )
        config, steps = smoke.call_args.args[2:]
        self.assertEqual((config.steps, config.batch_per_domain, steps), (6000, 8, 7))

    def test_training_accepts_all_explicit_hyperparameters(self):
        with (
            patch("cue_routing.cli.train", return_value={}) as training,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                ROOT,
                [
                    "train",
                    "--run-dir",
                    "unused",
                    "--method",
                    "rsc",
                    "--seed",
                    "42",
                    "--learning-rate",
                    "0.0002",
                    "--no-freeze-bn",
                    "--precision",
                    "bf16",
                ],
            )
        config = training.call_args.args[0]
        self.assertEqual((config.method, config.seed), ("rsc", 42))
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertFalse(config.freeze_bn)
        self.assertEqual(config.precision, "bf16")

    def test_matrix_rejects_singular_seed_instead_of_ignoring_it(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(ROOT, ["matrix", "--seed", "42"])


if __name__ == "__main__":
    unittest.main()
