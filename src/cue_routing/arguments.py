"""Reusable explicit CLI overrides for validated dataclass experiment configurations."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path


def add_config_arguments(parser, config_type, *, exclude=(), choices=None):
    """Expose scalar configuration fields without replacing omitted JSON settings."""
    defaults = config_type()
    choices = choices or {}
    for field in fields(config_type):
        if field.name in exclude:
            continue
        default = getattr(defaults, field.name)
        options = {
            "default": None,
            "help": f"Override {field.name} from --config (built-in default: {default!r})",
        }
        if type(default) is bool:
            options["action"] = argparse.BooleanOptionalAction
        else:
            if type(default) not in (str, int, float):
                raise TypeError(f"Unsupported CLI field type for {field.name}")
            options["type"] = type(default)
            if field.name in choices:
                options["choices"] = choices[field.name]
        parser.add_argument("--" + field.name.replace("_", "-"), **options)


def config_overrides(arguments, config_type, *, exclude=()):
    """Collect only explicitly supplied fields, retaining false, zero, and empty strings."""
    return {
        field.name: getattr(arguments, field.name)
        for field in fields(config_type)
        if field.name not in exclude and getattr(arguments, field.name, None) is not None
    }


def load_config_arguments(arguments, config_type, *, preset=None, exclude=()):
    """Merge JSON, command presets, and explicit flags before validating the final recipe."""
    config_path = getattr(arguments, "config", None)
    values = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else {}
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    values.update(preset or {})
    values.update(config_overrides(arguments, config_type, exclude=exclude))
    return config_type.from_dict(values)


def config_argument_tokens(arguments, config_type, *, exclude=()):
    """Forward explicit settings to another CLI without injecting dataclass defaults."""
    tokens = []
    for name, value in config_overrides(arguments, config_type, exclude=exclude).items():
        flag = "--" + name.replace("_", "-")
        if type(value) is bool:
            tokens.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            tokens.extend([flag, str(value)])
    return tokens


def load_data_smoke_config(arguments, config_type):
    """Keep smoke step counts separate and apply CPU defaults before explicit overrides."""
    device = getattr(arguments, "device", None) or "cpu"
    preset = {"device": device}
    if device == "cpu":
        preset.update(
            precision="fp32",
            batch_per_domain=1,
            eval_batch_size=3,
            steps=1,
            eval_every=1,
            workers=0,
            debug=True,
        )
    config = load_config_arguments(arguments, config_type, preset=preset)
    if not config.pretrained or config.image_size != 224:
        raise ValueError("data-smoke requires pretrained weights and 224px images")
    steps = getattr(arguments, "smoke_steps", None)
    if steps is None:
        steps = 10 if config.device == "cuda" else 1
    if type(steps) is not int or steps < 1:
        raise ValueError("data-smoke steps must be a positive integer")
    return config, steps
