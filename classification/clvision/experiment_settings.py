"""User-controlled recipes, CUE search grids, execution order, and optional time budgets."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import product

from .config import Config

CUE_FIELDS = frozenset({"gate_tau", "gate_temperature", "joint_floor"})


def deadline_time(value):
    """Resolve an optional timezone-qualified ISO deadline without using the host timezone."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("finish_by must be an ISO timestamp string or None")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("finish_by must include an explicit timezone, for example +09:00")
    timestamp = result.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("finish_by must resolve to a finite timestamp")
    return timestamp


@dataclass(frozen=True)
class Budget:
    """Reserve final work under an optional deadline; None permits every registered candidate."""

    finish_by: str | None = None
    reserve_minutes: float = 20.0
    time_safety_factor: float = 1.25

    def validate(self):
        """Validate numeric timing controls independently of the model's learning settings."""
        deadline_time(self.finish_by)
        for name, minimum in (("reserve_minutes", 0), ("time_safety_factor", 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"Invalid {name}")
        return self

    def deadline_time(self):
        """Return the validated absolute deadline or None for an unlimited search."""
        self.validate()
        return deadline_time(self.finish_by)

    def to_dict(self):
        """Record timing choices in the immutable experiment plan."""
        self.validate()
        return {
            "finish_by": self.finish_by,
            "reserve_minutes": self.reserve_minutes,
            "time_safety_factor": self.time_safety_factor,
        }


@dataclass(frozen=True)
class Search:
    """Register ordered CUE-only overrides while inheriting the explicit common recipe."""

    candidates: tuple | list
    metric: str = "accuracy"

    def validate(self):
        """Reject unknown tuning fields, invalid values, and unimplemented selection metrics."""
        if self.metric not in ("accuracy", "nll"):
            raise ValueError("Search metric must be accuracy or nll")
        if not isinstance(self.candidates, (tuple, list)) or not self.candidates:
            raise ValueError("Register at least one CUE candidate; use [{}] for a fixed CUE recipe")
        for candidate in self.candidates:
            if not isinstance(candidate, dict) or set(candidate) - CUE_FIELDS:
                raise ValueError(
                    "Candidates may override only gate_tau, gate_temperature, joint_floor"
                )
            Config.from_dict({"method": "cue", **candidate})
        return self

    def resolve(self, base):
        """Build complete matched CUE recipes without fixing optimizer, precision, or schedule values."""
        self.validate()
        if not isinstance(base, Config):
            raise ValueError("Common training settings must be a Config")
        base.validate()
        if base.method != "erm" or base.dataset not in ("cifar100", "tiny_imagenet"):
            raise ValueError("The common recipe must be ERM on CIFAR-100 or TinyImageNet")
        resolved = []
        for candidate in self.candidates:
            recipe = Config.from_dict({**base.to_dict(), **candidate, "method": "cue"})
            if recipe in resolved:
                raise ValueError("Duplicate resolved CUE candidate")
            resolved.append(recipe)
        return resolved

    def to_dict(self):
        """Snapshot the registered candidate order and validation selection rule."""
        self.validate()
        return {"candidates": [dict(row) for row in self.candidates], "metric": self.metric}


def cue_grid(**values):
    """Expand values in keyword order; omitted CUE fields inherit the common recipe."""
    if set(values) - CUE_FIELDS:
        raise ValueError("CUE grids may contain only gate_tau, gate_temperature, joint_floor")
    for name, choices in values.items():
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError(f"Grid field {name} requires a nonempty sequence")
    rows = [dict(zip(values, choices)) for choices in product(*values.values())]
    Search(rows).validate()
    return rows


def select_candidate(rows, metric="accuracy"):
    """Select final-epoch validation scores, breaking ties by the other metric and registered order."""
    if metric not in ("accuracy", "nll"):
        raise ValueError("Search metric must be accuracy or nll")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("Selection requires completed validation candidates")
    for row in rows:
        validation = row.get("validation") if isinstance(row, dict) else None
        if not isinstance(validation, dict) or any(
            type(validation.get(key)) not in (int, float) or not math.isfinite(validation[key])
            for key in ("accuracy", "nll")
        ):
            raise ValueError("Selection requires finite final-epoch validation accuracy and nll")
    return min(
        enumerate(rows),
        key=lambda item: (
            (-item[1]["validation"]["accuracy"], item[1]["validation"]["nll"], item[0])
            if metric == "accuracy"
            else (item[1]["validation"]["nll"], -item[1]["validation"]["accuracy"], item[0])
        ),
    )[1]


def validate_stages(stages, *, include_baseline=True):
    """Keep preflight first and final evaluation last, with an optional matched baseline stage."""
    if not isinstance(stages, (tuple, list)):
        raise ValueError("Stages must be an ordered tuple or list")
    order = tuple(stages)
    if not include_baseline:
        if order != ("preflight", "tune", "evaluate"):
            raise ValueError("CUE-only search requires preflight, tune, evaluate")
        return order
    if (
        len(order) != 4
        or order[0] != "preflight"
        or order[-1] != "evaluate"
        or order[1:3] not in (("tune", "baseline"), ("baseline", "tune"))
    ):
        raise ValueError("Use preflight, then tune/baseline in either order, then evaluate")
    return order
