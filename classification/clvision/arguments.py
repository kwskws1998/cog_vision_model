"""Shared study inputs for the experiment CLI and the fail-fast stage driver."""

from pathlib import Path

from .display import add_display_arguments
from .settings import StudySettings

TRAINING_FIELDS = {
    "tiny_imagenet_size": int,
    "batch_size": int,
    "eval_batch_size": int,
    "learning_rate": float,
    "momentum": float,
    "weight_decay": float,
    "lr_decay": float,
    "gate_temperature": float,
    "joint_floor": float,
    "zipf_aux_weight": float,
    "zipf_power": float,
    "label_smoothing": float,
    "max_vram_gib": float,
    "gradient_clip": float,
}


def add_study_arguments(parser, root):
    """Expose explicit command-line overrides without replacing values omitted by the user."""
    parser.add_argument(
        "--config", type=Path, default=Path(root) / "classification/configs/experiment.json"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Set both main and mechanism seeds")
    parser.add_argument("--main-seeds", type=int, nargs="+")
    parser.add_argument("--mechanism-seeds", type=int, nargs="+")
    parser.add_argument("--development-seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--eval-precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--epochs", type=int, help="Natural-image training epochs")
    parser.add_argument("--milestones", type=int, nargs="*", help="Natural-image LR drop epochs")
    parser.add_argument("--mechanism-epochs", type=int)
    parser.add_argument("--mechanism-milestones", type=int, nargs="*")
    for name, value_type in TRAINING_FIELDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=value_type)
    add_display_arguments(parser)


def load_study_arguments(arguments):
    """Merge file settings with explicit CLI values, including dataset-specific precedence."""
    import json

    values = json.loads(Path(arguments.config).read_text())
    if not isinstance(values, dict):
        raise ValueError("Study configuration must be a JSON object")
    for name in ("training", "natural_schedule", "mechanism_schedule", "dataset_overrides"):
        if name in values and not isinstance(values[name], dict):
            raise ValueError(f"{name} must be a JSON object")
    if any(not isinstance(item, dict) for item in values.get("dataset_overrides", {}).values()):
        raise ValueError("Each dataset_overrides entry must be a JSON object")
    seeds = getattr(arguments, "seeds", None)
    for name in ("main_seeds", "mechanism_seeds"):
        explicit = getattr(arguments, name, None)
        if explicit is not None or seeds is not None:
            values[name] = explicit if explicit is not None else seeds
    for name in ("development_seed", "workers"):
        value = getattr(arguments, name, None)
        if value is not None:
            values[name] = value
    updates = {
        name: getattr(arguments, name)
        for name in (*TRAINING_FIELDS, "device", "precision", "eval_precision")
        if getattr(arguments, name, None) is not None
    }
    values["training"] = {**values.get("training", {}), **updates}
    for scope in ("natural_schedule", "mechanism_schedule"):
        if scope in values:
            values[scope] = {
                key: value for key, value in values[scope].items() if key not in updates
            }
    for dataset, overrides in values.get("dataset_overrides", {}).items():
        values["dataset_overrides"][dataset] = {
            key: value for key, value in overrides.items() if key not in updates
        }
    for scope, prefix in (("natural_schedule", ""), ("mechanism_schedule", "mechanism_")):
        schedule = {
            name: getattr(arguments, prefix + name)
            for name in ("epochs", "milestones")
            if getattr(arguments, prefix + name, None) is not None
        }
        values[scope] = {**values.get(scope, {}), **schedule}
        datasets = ("dominoes",) if prefix else ("cifar100", "tiny_imagenet")
        for dataset in datasets:
            if dataset in values.get("dataset_overrides", {}):
                values["dataset_overrides"][dataset] = {
                    key: value
                    for key, value in values["dataset_overrides"][dataset].items()
                    if key not in schedule
                }
    return StudySettings.from_dict(values)


def study_argument_tokens(arguments):
    """Forward only explicit inputs so both command entry points resolve an identical study."""
    result = ["--config", str(Path(arguments.config).resolve())]
    names = (
        "seeds",
        "main_seeds",
        "mechanism_seeds",
        "development_seed",
        "workers",
        "device",
        "precision",
        "eval_precision",
        "epochs",
        "milestones",
        "mechanism_epochs",
        "mechanism_milestones",
        "progress",
        "logging_steps",
        "progress_interval",
        *TRAINING_FIELDS,
    )
    for name in names:
        value = getattr(arguments, name, None)
        if value is not None:
            result.append("--" + name.replace("_", "-"))
            result.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return result


def study_preview(settings):
    """Resolve full recipes and counts without opening datasets, downloading, or training."""
    from .config import MAIN_METHODS, MECHANISM_METHODS
    from .workflow import base_recipe, candidate_configs

    settings = settings.validate()
    resolved = settings.to_dict()
    recipes = {
        dataset: base_recipe(dataset, settings)
        for dataset in ("cifar100", "tiny_imagenet", "dominoes")
    }
    development = {
        dataset: len(candidate_configs(recipes[dataset], settings))
        for dataset in ("cifar100", "tiny_imagenet")
    }
    counts = {
        "development": sum(development.values()),
        "main": 2 * len(MAIN_METHODS) * len(resolved["main_seeds"]),
        "mechanism": len(MECHANISM_METHODS) * len(resolved["mechanism_seeds"]),
    }
    return {
        "settings": resolved,
        "resolved_recipes": {name: config.to_dict() for name, config in recipes.items()},
        "development_runs_by_dataset": development,
        "training_runs": {**counts, "total": sum(counts.values())},
        "training_performed": False,
        "final_evaluation_performed": False,
    }
