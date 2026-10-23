"""Bounded validation-only tuning and matched ERM/CUE experiment inputs."""

import argparse
from dataclasses import dataclass, replace

from .cue_settings import CueSettings, add_cue_arguments, load_cue_arguments

METHODS = ("erm", "cue", "fixed_global_local")


@dataclass(frozen=True)
class AblationSettings(CueSettings):
    """Tune a shared baseline recipe, then intervention coefficients, before paired seeds."""

    datasets: tuple = ("cifar100",)
    methods: tuple = ("erm", "cue")
    development_seed: int = 2027
    tune: bool = True
    lr_candidates: tuple = (0.1, 0.4, 0.8)
    tau_candidates: tuple = (0.05, 0.15, 0.3)
    fixed_candidates: tuple = (0.1, 0.25, 0.5)

    @classmethod
    def from_dict(cls, value):
        """Reject overlapping seed roles, invalid candidate recipes, and undeclared methods."""
        if not isinstance(value, dict):
            raise ValueError("Ablation settings must be an object")
        value = dict(value)
        for name in ("methods", "lr_candidates", "tau_candidates", "fixed_candidates"):
            if name in value:
                if not isinstance(value[name], (list, tuple)):
                    raise ValueError(f"{name} must be a list")
                value[name] = tuple(value[name])
        result = super().from_dict(value)
        if (
            set(result.methods) - set(METHODS)
            or not {"erm", "cue"}.issubset(result.methods)
            or len(set(result.methods)) != len(result.methods)
        ):
            raise ValueError("Ablation methods must include unique erm and cue arms")
        if type(result.tune) is not bool:
            raise ValueError("tune must be a boolean")
        if (
            type(result.development_seed) is not int
            or not 0 <= result.development_seed < 2**63
            or result.development_seed in result.seeds
        ):
            raise ValueError("Development seed must be valid and separate from final seeds")
        for values, key in (
            (result.lr_candidates, "learning_rate"),
            (result.tau_candidates, "gate_tau"),
            (result.fixed_candidates, "fixed_lambda"),
        ):
            if (
                not values
                or any(type(item) not in (int, float) for item in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Use nonempty unique numerical candidates")
            for dataset in result.datasets:
                for item in values:
                    replace(result.recipe(dataset), **{key: item}).validate()
        return result

    def trials(self, dataset, shared_lr=None):
        """Resolve conditional intervention trials after ERM selects the common learning rate."""
        if not self.tune:
            return []
        base = self.recipe(dataset, self.development_seed)
        trials = [
            (f"erm_lr{index}", replace(base, method="erm", learning_rate=value))
            for index, value in enumerate(self.lr_candidates)
        ]
        base = replace(base, learning_rate=base.learning_rate if shared_lr is None else shared_lr)
        trials += [
            (f"cue_tau{index}", replace(base, method="cue", gate_tau=value))
            for index, value in enumerate(self.tau_candidates)
        ]
        if "fixed_global_local" in self.methods:
            trials += [
                (
                    f"fixed_weight{index}",
                    replace(base, method="fixed_global_local", fixed_lambda=value),
                )
                for index, value in enumerate(self.fixed_candidates)
            ]
        return [(name, config.validate()) for name, config in trials]

    def preview(self):
        """Count all tuning and confirmation jobs, including optional fixed-consistency control."""
        trials = {name: self.trials(name) for name in self.datasets}
        development = sum(len(rows) for rows in trials.values())
        main = len(self.datasets) * len(self.methods) * len(self.seeds)
        return {
            "mode": "matched_ablation",
            "methods": list(self.methods),
            "datasets": list(self.datasets),
            "development_seed": self.development_seed,
            "seeds": list(self.seeds),
            "development_runs": development,
            "main_runs": main,
            "training_runs": development + main,
            "training_epochs": sum(
                self.recipe(name).epochs * (len(trials[name]) + len(self.methods) * len(self.seeds))
                for name in self.datasets
            ),
            "corruption_cells": 0,
            "selection": "ERM validation selects shared LR; intervention validation selects coefficients"
            if self.tune
            else "fixed_before_training",
            "settings": self.to_dict(),
            "benchmark_evidence": not self.debug,
        }


def add_ablation_arguments(parser, root):
    """Share training flags while exposing bounded search grids and an optional mechanism control."""
    add_cue_arguments(parser, root)
    parser.set_defaults(
        config=root / "classification/configs/ablation_5090.json",
        runs_dir=root / "runs/ablation_5090",
    )
    parser.add_argument("--methods", choices=METHODS, nargs="+")
    parser.add_argument("--development-seed", type=int)
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=None)
    for name in ("lr_candidates", "tau_candidates", "fixed_candidates"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, nargs="+")


def load_ablation_arguments(arguments):
    """Avoid silently ignoring scalar overrides when the user requested a search grid."""
    settings = load_cue_arguments(arguments, AblationSettings)
    if settings.tune and any(
        getattr(arguments, name, None) is not None for name in ("learning_rate", "gate_tau")
    ):
        raise ValueError(
            "Use --lr-candidates / --tau-candidates for tuning, or --no-tune with scalar overrides"
        )
    return settings
