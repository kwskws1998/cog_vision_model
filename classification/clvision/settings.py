"""Editable study inputs resolved once into validated, reproducible training recipes."""

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from .config import DATASETS, Config, recipe

NATURAL_DATASETS = ("cifar100", "tiny_imagenet")
CANDIDATE_FIELDS = {
    "cue": "gate_tau",
    "fixed_global_local": "fixed_lambda",
    "zipfls_dense2": "zipf_lambda",
}
TUNED_FIELDS = set(CANDIDATE_FIELDS.values())
TRAINING_FIELDS = (
    {item.name for item in fields(Config)}
    - TUNED_FIELDS
    - {
        "dataset",
        "method",
        "seed",
        "workers",
        "debug",
        "debug_max_batches",
    }
)
SCHEDULE_FIELDS = {"epochs", "milestones", "lr_decay"}


def default_candidates():
    """Return fresh per-dataset grids with equal validation budgets across interventions."""
    return {
        dataset: {
            "cue": [0.05, 0.15, 0.3],
            "fixed_global_local": [0.1, 0.25, 0.5],
            "zipfls_dense2": [0.03, 0.1, 0.3] if dataset == "cifar100" else [0.3, 1.0, 3.0],
        }
        for dataset in NATURAL_DATASETS
    }


def checked_overrides(value, allowed, name):
    """Validate override keys early while preserving explicit values for Config validation."""
    if isinstance(value, dict) and set(value) & TUNED_FIELDS:
        tuned = sorted(set(value) & TUNED_FIELDS)
        raise ValueError(
            f"{name} cannot set validation-selected fields {tuned}; "
            "set development_candidates instead, with equal candidate counts across methods"
        )
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"{name} must be an object containing only {sorted(allowed)}")
    result = dict(value)
    if "milestones" in result:
        if not isinstance(result["milestones"], (list, tuple)):
            raise ValueError(f"{name}.milestones must be a list")
        result["milestones"] = tuple(result["milestones"])
    return result


@dataclass(frozen=True)
class StudySettings:
    """Separate study scope, shared training settings, dataset schedules, and validation grids."""

    debug: bool = False
    workers: int = 4
    main_seeds: tuple = (0, 1)
    mechanism_seeds: tuple = (0, 1)
    development_seed: int = 2027
    training: dict = field(default_factory=dict)
    natural_schedule: dict = field(default_factory=dict)
    mechanism_schedule: dict = field(default_factory=dict)
    dataset_overrides: dict = field(default_factory=dict)
    development_candidates: dict = field(default_factory=default_candidates)

    def validate(self):
        """Reject ambiguous scope, unequal tuning budgets, and invalid resolved recipes."""
        if type(self.debug) is not bool or type(self.workers) is not int or self.workers < 0:
            raise ValueError("debug must be boolean and workers must be a nonnegative integer")
        for name in ("main_seeds", "mechanism_seeds"):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or not values
                or any(type(value) is not int or not 0 <= value < 2**63 for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"{name} must contain unique nonnegative integer seeds below 2**63"
                )
        if type(self.development_seed) is not int or not 0 <= self.development_seed < 2**63:
            raise ValueError("development_seed must be a nonnegative integer below 2**63")
        checked_overrides(self.training, TRAINING_FIELDS, "training")
        checked_overrides(self.natural_schedule, SCHEDULE_FIELDS, "natural_schedule")
        checked_overrides(self.mechanism_schedule, SCHEDULE_FIELDS, "mechanism_schedule")
        if not isinstance(self.dataset_overrides, dict) or set(self.dataset_overrides) - set(
            DATASETS
        ):
            raise ValueError("dataset_overrides contains an unknown dataset")
        for dataset, updates in self.dataset_overrides.items():
            checked_overrides(updates, TRAINING_FIELDS, f"dataset_overrides.{dataset}")
        grids = self.development_candidates
        if not isinstance(grids, dict) or set(grids) != set(NATURAL_DATASETS):
            raise ValueError("development_candidates requires cifar100 and tiny_imagenet grids")
        for dataset, methods in grids.items():
            if not isinstance(methods, dict) or set(methods) != set(CANDIDATE_FIELDS):
                raise ValueError(
                    f"{dataset} requires candidate grids for all three tunable methods"
                )
            counts = []
            base = self.recipe(dataset)
            for method, values in methods.items():
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError("Each development candidate grid must be nonempty")
                if any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                ):
                    raise ValueError("Development candidates must be finite numbers")
                if len(set(values)) != len(values):
                    raise ValueError("Development candidates must be unique within each method")
                counts.append(len(values))
                for value in values:
                    replace(base, method=method, **{CANDIDATE_FIELDS[method]: value}).validate()
            if len(set(counts)) != 1:
                raise ValueError("All tunable methods require equal development candidate counts")
        for dataset in DATASETS:
            self.recipe(dataset)
        return self

    def recipe(self, dataset):
        """Resolve shared, schedule, and dataset overrides before applying explicit smoke limits."""
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        schedule = self.mechanism_schedule if dataset == "dominoes" else self.natural_schedule
        updates = checked_overrides(self.training, TRAINING_FIELDS, "training")
        updates.update(checked_overrides(schedule, SCHEDULE_FIELDS, "schedule"))
        updates.update(
            checked_overrides(self.dataset_overrides.get(dataset, {}), TRAINING_FIELDS, dataset)
        )
        updates["workers"] = self.workers
        resolved = recipe(dataset, debug=self.debug, **updates)
        if self.debug:
            return replace(
                resolved,
                debug=True,
                device="cpu",
                precision="fp32",
                eval_precision="fp32",
                epochs=2,
                warmup_epochs=0,
                milestones=(1,),
                batch_size=2,
                eval_batch_size=4,
                debug_max_batches=2,
            ).validate()
        return resolved

    def to_dict(self):
        """Return detached portable JSON values suitable for CLI merging and saved contracts."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        """Fill omitted fields, accept legacy debug/workers settings, and reject unknown keys."""
        if not isinstance(value, dict):
            raise ValueError("Study settings must be a JSON object")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown study settings: {sorted(unknown)}")
        values = dict(value)
        for key in ("main_seeds", "mechanism_seeds"):
            if key not in values:
                values[key] = [0] if values.get("debug", False) else [0, 1]
            if not isinstance(values[key], (list, tuple)):
                raise ValueError(f"{key} must be a list")
            values[key] = tuple(values[key])
        if "development_candidates" in values:
            overrides = values["development_candidates"]
            if not isinstance(overrides, dict) or set(overrides) - set(NATURAL_DATASETS):
                raise ValueError("development_candidates contains an unknown dataset")
            grids = default_candidates()
            for dataset, methods in overrides.items():
                if not isinstance(methods, dict) or set(methods) - set(CANDIDATE_FIELDS):
                    raise ValueError(f"Invalid development methods for {dataset}")
                grids[dataset].update(methods)
            values["development_candidates"] = grids
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        """Read editable experiment inputs independently of frozen per-run Config files."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def resolve_settings(settings=None, *, debug=None, workers=None):
    """Normalize public inputs without replacing explicitly supplied seeds or training fields."""
    if isinstance(settings, StudySettings):
        values = settings.to_dict()
    elif settings is None:
        values = {}
    elif isinstance(settings, dict):
        values = dict(settings)
    else:
        raise ValueError("settings must be StudySettings or a dictionary")
    if debug is not None:
        values["debug"] = debug
    if workers is not None:
        values["workers"] = workers
    return StudySettings.from_dict(values)
