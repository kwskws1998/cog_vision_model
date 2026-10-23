"""Selectable residual blocks across config precedence, public CLIs, and smoke entrypoints."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clvision.ablation_settings import AblationSettings, add_ablation_arguments
from clvision.arguments import add_study_arguments, load_study_arguments
from clvision.cli import main
from clvision.config import Config
from clvision.cue_settings import CueSettings, ErmSettings, add_cue_arguments, load_cue_arguments
from clvision.settings import StudySettings

ROOT = Path(__file__).resolve().parents[2]


def load_driver():
    """Import the stage wrapper without executing its main function."""
    specification = importlib.util.spec_from_file_location(
        "activation_driver", ROOT / "classification/run.py"
    )
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    return driver


class ActivationSettingsTests(unittest.TestCase):
    """Require architecture choices to survive every public resolution and forwarding route."""

    def test_omitted_activation_keeps_postact_and_both_choices_round_trip(self):
        """Legacy JSON remains post-activation and new recipes persist their explicit choice."""
        self.assertEqual(Config.from_dict({}).activation, "postact")
        for activation in ("postact", "preact"):
            config = Config(activation=activation).validate()
            self.assertEqual(Config.from_dict(config.to_dict()), config)
            for settings_type in (StudySettings, CueSettings, ErmSettings, AblationSettings):
                with self.subTest(activation=activation, settings_type=settings_type):
                    settings = settings_type.from_dict({"training": {"activation": activation}})
                    self.assertEqual(settings.recipe("cifar100").activation, activation)
                    self.assertEqual(settings.recipe("tiny_imagenet").activation, activation)

    def test_invalid_activation_rejected_in_all_json_scopes(self):
        """Typos and non-string architecture values fail before data or model creation."""
        for value in ("pre", "post", "PreAct", "relu", "", None, True, 1, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "activation"):
                Config(activation=value).validate()
            for settings_type in (StudySettings, CueSettings, ErmSettings, AblationSettings):
                for fields in (
                    {"training": {"activation": value}},
                    {"dataset_overrides": {"tiny_imagenet": {"activation": value}}},
                ):
                    with (
                        self.subTest(value=value, settings_type=settings_type, fields=fields),
                        self.assertRaisesRegex(ValueError, "activation"),
                    ):
                        settings_type.from_dict(fields)

    def test_json_dataset_precedence_and_explicit_cli_override(self):
        """An omitted flag preserves per-dataset choices; an explicit flag overrides both scopes."""
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "architecture choices.json"
            config_path.write_text(
                json.dumps(
                    {
                        "training": {"activation": "preact"},
                        "dataset_overrides": {"tiny_imagenet": {"activation": "postact"}},
                    }
                )
            )
            for settings_type in (StudySettings, CueSettings, ErmSettings, AblationSettings):
                parser = argparse.ArgumentParser()
                if settings_type is StudySettings:
                    add_study_arguments(parser, ROOT)
                elif settings_type is AblationSettings:
                    add_ablation_arguments(parser, ROOT)
                else:
                    add_cue_arguments(parser, ROOT)
                for override in (None, "postact", "preact"):
                    tokens = ["--config", str(config_path)]
                    if override is not None:
                        tokens.extend(("--activation", override))
                    arguments = parser.parse_args(tokens)
                    settings = (
                        load_study_arguments(arguments)
                        if settings_type is StudySettings
                        else load_cue_arguments(arguments, settings_type)
                    )
                    with self.subTest(settings_type=settings_type, override=override):
                        self.assertEqual(
                            settings.recipe("cifar100").activation, override or "preact"
                        )
                        self.assertEqual(
                            settings.recipe("tiny_imagenet").activation, override or "postact"
                        )
                        if settings_type is AblationSettings:
                            self.assertTrue(
                                all(
                                    recipe.activation == (override or "preact")
                                    for _, recipe in settings.trials("cifar100")
                                )
                            )

    def test_both_entrypoints_preview_each_training_mode(self):
        """Driver and direct CLI previews report the same selected architecture without data."""
        driver = load_driver()
        for entry in ("run.py", "experiment.py"):
            for command in ("cue", "erm", "ablation", "plan", "train"):
                direct_command = (
                    "study" if command == "train" and entry == "experiment.py" else command
                )
                tokens = [entry, direct_command, "--activation", "preact"]
                if command != "plan":
                    tokens.append("--dry-run")
                output = io.StringIO()
                with (
                    self.subTest(entry=entry, command=command),
                    patch("sys.argv", tokens),
                    patch("clvision.cli.load_manifest", side_effect=AssertionError("no data")),
                    contextlib.redirect_stdout(output),
                ):
                    driver.main() if entry == "run.py" else main(ROOT)
                preview = json.loads(output.getvalue())
                if command == "ablation":
                    settings = AblationSettings.from_dict(preview["settings"])
                    recipes = [settings.recipe(dataset).to_dict() for dataset in settings.datasets]
                else:
                    recipes = preview["resolved_recipes"].values()
                self.assertTrue(all(recipe["activation"] == "preact" for recipe in recipes))

    def test_driver_forwards_selected_activation_and_custom_config(self):
        """All subprocess routes preserve flags and paths, including the separate smoke command."""
        driver = load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "custom settings.json"
            config_path.write_text("{}")
            for command in ("train", "cue", "erm", "ablation", "smoke"):
                tokens = ["run.py", command, "--activation", "preact"]
                if command != "smoke":
                    tokens.extend(("--config", str(config_path)))
                with (
                    self.subTest(command=command),
                    patch("sys.argv", tokens),
                    patch.object(driver.subprocess, "run") as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    driver.main()
                forwarded = run.call_args.args[0]
                self.assertEqual(forwarded[forwarded.index("--activation") + 1], "preact")
                if command != "smoke":
                    self.assertEqual(
                        Path(forwarded[forwarded.index("--config") + 1]).resolve(),
                        config_path.resolve(),
                    )

    def test_direct_profile_and_smoke_dispatch_selected_activation(self):
        """Diagnostic entrypoints use the requested blocks rather than their default backbone."""
        with (
            patch("sys.argv", ["experiment.py", "smoke", "--activation", "preact"]),
            patch("clvision.cli.smoke", return_value={}) as smoke,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(ROOT)
        self.assertEqual(smoke.call_args.args, (ROOT, "preact"))
        manifest = {"dataset": "cifar100", "dataset_id": "fixture"}
        for command in ("data-smoke", "profile"):
            with (
                self.subTest(command=command),
                patch(
                    "sys.argv",
                    ["experiment.py", command, "--dataset", "cifar100", "--activation", "preact"],
                ),
                patch("clvision.cli.load_manifest", return_value=manifest),
                patch("clvision.cli.data_smoke", return_value={}) as data_smoke,
                patch(
                    "clvision.cli.profile_dataset",
                    return_value={"measured": True, "passes_budget": True},
                ) as profile,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(ROOT)
            if command == "data-smoke":
                self.assertEqual(data_smoke.call_args.args[-1], "preact")
            else:
                self.assertEqual(profile.call_args.args[2].activation, "preact")

    def test_public_cli_rejects_unknown_activation(self):
        """An invalid command-line value never silently falls back to post-activation."""
        for command in ("cue", "erm", "ablation", "study", "profile", "smoke", "data-smoke"):
            with (
                self.subTest(command=command),
                patch("sys.argv", ["experiment.py", command, "--activation", "pre"]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(ROOT)
            self.assertEqual(error.exception.code, 2)

    def test_shipped_presets_preserve_postact(self):
        """Updating code does not silently replace the backbone in existing experiment presets."""
        for name in ("experiment", "cue_5090", "erm_5090", "ablation_5090", "tiny32_ablation_5090"):
            with self.subTest(name=name):
                values = json.loads((ROOT / "classification/configs" / f"{name}.json").read_text())
                self.assertEqual(values["training"]["activation"], "postact")
