"""Editable CUE-only recipes and explicit command-line overrides for a single GPU."""

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .config import Config

DATASETS = ("cifar100", "tiny_imagenet")
TRAINING_FIELDS = {item.name for item in fields(Config)} - {
    "dataset",
    "method",
    "seed",
    "debug",
    "debug_max_batches",
    "fixed_lambda",
    "zipf_lambda",
    "zipf_aux_weight",
    "zipf_power",
    "label_smoothing",
}


@dataclass(frozen=True)
class CueSettings:
    """Freeze the selected datasets, seeds, and fixed CUE settings without a search grid."""

    datasets: tuple = DATASETS
    seeds: tuple = (0, 1)
    training: dict = field(default_factory=dict)
    dataset_overrides: dict = field(default_factory=dict)
    debug: bool = False

    @classmethod
    def from_dict(cls, value):
        """Reject unknown inputs and validate each resolved recipe before touching datasets."""
        if not isinstance(value, dict) or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("Unknown CUE study setting")
        value = dict(value)
        for name in ("datasets", "seeds"):
            if name in value:
                if not isinstance(value[name], (list, tuple)):
                    raise ValueError(f"{name} must be a list")
                value[name] = tuple(value[name])
        result = cls(**value)
        if (
            not result.datasets
            or any(type(name) is not str or name not in DATASETS for name in result.datasets)
            or len(set(result.datasets)) != len(result.datasets)
        ):
            raise ValueError("Choose unique natural-image datasets")
        if (
            not result.seeds
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in result.seeds)
            or len(set(result.seeds)) != len(result.seeds)
        ):
            raise ValueError("Choose nonempty unique nonnegative seeds")
        if type(result.debug) is not bool:
            raise ValueError("debug must be a boolean")
        if not isinstance(result.dataset_overrides, dict) or set(result.dataset_overrides) - set(
            DATASETS
        ):
            raise ValueError("Unknown dataset override")
        for override in (result.training, *result.dataset_overrides.values()):
            if not isinstance(override, dict) or set(override) - TRAINING_FIELDS:
                raise ValueError("Unknown or forbidden CUE training setting")
        for dataset in DATASETS:
            result.recipe(dataset)
        return result

    def recipe(self, dataset, seed=None):
        """Resolve common inputs and dataset overrides; debug runs use small explicit fixtures."""
        if dataset not in DATASETS:
            raise ValueError("CUE-only mode supports natural-image datasets")
        value = {
            **self.training,
            **self.dataset_overrides.get(dataset, {}),
            "dataset": dataset,
            "method": "cue",
            "seed": self.seeds[0] if seed is None else seed,
        }
        Config.from_dict({**value, "debug": self.debug})
        if self.debug:
            value.update(
                debug=True,
                epochs=2,
                milestones=[1],
                warmup_epochs=0,
                batch_size=2,
                eval_batch_size=4,
                workers=0,
                device="cpu",
                precision="fp32",
                eval_precision="fp32",
                debug_max_batches=1,
            )
        return Config.from_dict(value)

    def to_dict(self):
        """Use JSON-native inputs in source-bound plans."""
        return json.loads(json.dumps(asdict(self)))

    def preview(self):
        """Show the exact run budget without loading data or initializing CUDA."""
        recipes = {name: self.recipe(name).to_dict() for name in self.datasets}
        return {
            "mode": "cue_only",
            "methods": ["cue"],
            "datasets": list(self.datasets),
            "seeds": list(self.seeds),
            "training_runs": len(self.datasets) * len(self.seeds),
            "training_epochs": sum(item["epochs"] for item in recipes.values()) * len(self.seeds),
            "development_runs": 0,
            "competitor_training_runs": 0,
            "corruption_cells": 0,
            "fixed_hyperparameters": True,
            "benchmark_evidence": not self.debug,
            "resolved_recipes": recipes,
            "reference_comparison": "published_scores_with_explicit_protocol_differences",
        }


def add_cue_arguments(parser, root):
    """Expose editable JSON plus familiar hyperparameter flags in both Python entrypoints."""
    parser.add_argument(
        "--config", type=Path, default=root / "classification/configs/cue_5090.json"
    )
    parser.add_argument("--data-root", type=Path, default=root / "data/classification")
    parser.add_argument("--runs-dir", type=Path, default=root / "runs/cue_5090")
    parser.add_argument("--datasets", choices=DATASETS, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    for name in ("epochs", "batch_size", "eval_batch_size", "workers", "warmup_epochs"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--milestones", type=int, nargs="*")
    for name in (
        "learning_rate",
        "momentum",
        "weight_decay",
        "lr_decay",
        "max_vram_gib",
        "gate_tau",
        "gate_temperature",
        "joint_floor",
        "gradient_clip",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--eval-precision", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")


def load_cue_arguments(arguments):
    """Give explicit CLI values precedence over both common and dataset-specific JSON values."""
    value = CueSettings.from_dict(json.loads(arguments.config.read_text())).to_dict()
    for name in ("datasets", "seeds", "debug"):
        if getattr(arguments, name) is not None:
            value[name] = getattr(arguments, name)
    for name in TRAINING_FIELDS:
        explicit = getattr(arguments, name, None)
        if explicit is not None:
            value["training"][name] = explicit
            for override in value["dataset_overrides"].values():
                override.pop(name, None)
    return CueSettings.from_dict(value)
