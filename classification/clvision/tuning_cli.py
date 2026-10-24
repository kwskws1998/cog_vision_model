"""Reusable command-line CUE searches; experiment values come from arguments, not recipe files."""

import argparse
import json
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from cue_routing.io import read_json

from .config import ACTIVATIONS, Config
from .display import add_display_arguments, display_options, options_from_arguments
from .experiment import Comparison
from .experiment_settings import Budget, Search, cue_grid, validate_stages

INTEGER_FIELDS = ("epochs", "batch_size", "eval_batch_size", "warmup_epochs", "workers")
FLOAT_FIELDS = (
    "learning_rate",
    "momentum",
    "weight_decay",
    "lr_decay",
    "gate_tau",
    "gate_temperature",
    "joint_floor",
    "gradient_clip",
    "max_vram_gib",
)
CHOICES = {
    "activation": ACTIVATIONS,
    "initialization": ("reference", "msra"),
    "normalization": ("dataset", "zipfls"),
    "precision": ("fp32", "fp16", "bf16"),
    "eval_precision": ("fp32", "fp16", "bf16"),
    "device": ("cuda", "cpu"),
    "tiny_imagenet_size": (32, 64),
}
GRID_FIELDS = ("gate_tau", "gate_temperature", "joint_floor")


def add_tuning_arguments(parser, root):
    """Expose common recipes, search space, execution order, references, and timing controls."""
    defaults = Config()
    parser.add_argument("--data-root", type=Path, default=root / "data/classification")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar100", "tiny_imagenet"), default="cifar100")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    for name in INTEGER_FIELDS + FLOAT_FIELDS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name in INTEGER_FIELDS else float,
            default=getattr(defaults, name),
        )
    for name, choices in CHOICES.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            choices=choices,
            type=int if name == "tiny_imagenet_size" else str,
            default=getattr(defaults, name),
        )
    parser.add_argument("--milestones", type=int, nargs="*", default=list(defaults.milestones))
    for name in GRID_FIELDS:
        parser.add_argument(
            "--tune-" + name.replace("_", "-"),
            type=float,
            nargs="+",
            help=f"Search values for {name}; omit to keep the scalar recipe value",
        )
    parser.add_argument("--search-strategy", choices=("grid", "random"), default="grid")
    parser.add_argument("--search-seed", type=int, default=2027)
    parser.add_argument("--max-trials", type=int, help="Maximum candidates per seed; default all")
    parser.add_argument("--selection-metric", choices=("accuracy", "nll"), default="accuracy")
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reference", type=Path, help="Verified validation reference for one seed")
    parser.add_argument(
        "--stages", nargs="+", choices=("preflight", "tune", "baseline", "evaluate")
    )
    parser.add_argument(
        "--finish-by", help="Optional deadline with timezone, e.g. 2026-10-25T10:30:00+09:00"
    )
    parser.add_argument("--reserve-minutes", type=float, default=20.0)
    parser.add_argument("--time-safety-factor", type=float, default=1.25)
    parser.add_argument("--preflight-steps", type=int, default=10)
    parser.add_argument(
        "--cpu-threads", type=int, help="PyTorch CPU threads; default preserves the runtime setting"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Synthetic two-epoch CPU check; no benchmark evidence"
    )
    parser.add_argument("--resume", action="store_true")
    add_display_arguments(parser)


def resolve_tuning_arguments(arguments, root):
    """Validate the entire invocation before data reads, directory creation, or CUDA work."""
    options_from_arguments(arguments)
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("--seeds must be nonempty and unique")
    if arguments.search_seed < 0 or arguments.search_seed >= 2**63:
        raise ValueError("--search-seed must be in [0, 2**63)")
    if arguments.max_trials is not None and arguments.max_trials < 1:
        raise ValueError("--max-trials must be positive")
    if arguments.cpu_threads is not None and arguments.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if arguments.reference is not None and (len(arguments.seeds) != 1 or arguments.smoke):
        raise ValueError("--reference requires one matching production seed; omit it for --smoke")
    stages = validate_stages(
        arguments.stages
        or (
            ("preflight", "tune", "baseline", "evaluate")
            if arguments.include_baseline
            else ("preflight", "tune", "evaluate")
        ),
        include_baseline=arguments.include_baseline,
    )
    values = {
        name: getattr(arguments, name) for name in INTEGER_FIELDS + FLOAT_FIELDS + tuple(CHOICES)
    }
    config = Config(
        **values,
        dataset=arguments.dataset,
        seed=arguments.seeds[0],
        method="erm" if arguments.include_baseline else "cue",
        milestones=tuple(arguments.milestones),
    )
    if arguments.smoke:
        config = replace(
            config,
            device="cpu",
            precision="fp32",
            eval_precision="fp32",
            debug=True,
            epochs=2,
            milestones=(1,),
            warmup_epochs=0,
            workers=0,
            batch_size=2,
            eval_batch_size=4,
            debug_max_batches=1,
        )
    config.validate()
    grid = cue_grid(
        **{
            name: getattr(arguments, "tune_" + name)
            for name in GRID_FIELDS
            if getattr(arguments, "tune_" + name) is not None
        }
    )
    Search(grid, arguments.selection_metric).resolve(replace(config, method="erm"))
    if arguments.search_strategy == "random":
        random.Random(arguments.search_seed).shuffle(grid)
    candidates = grid if arguments.max_trials is None else grid[: arguments.max_trials]
    budget = Budget(arguments.finish_by, arguments.reserve_minutes, arguments.time_safety_factor)
    budget.validate()
    reference = None if arguments.reference is None else read_json(arguments.reference)
    directory = arguments.runs_dir.resolve()
    data_root = directory / "synthetic_data" if arguments.smoke else arguments.data_root.resolve()
    runs = [
        Comparison(
            root=root,
            data_root=data_root,
            directory=directory if len(arguments.seeds) == 1 else directory / f"seed{seed}",
            config=replace(config, seed=seed),
            search=Search(candidates, arguments.selection_metric),
            budget=budget,
            resume=arguments.resume,
            preflight_steps=arguments.preflight_steps,
            include_baseline=arguments.include_baseline,
            reference=reference,
        )
        for seed in arguments.seeds
    ]
    preview = {
        "runs_dir": str(directory),
        "data_root": str(data_root),
        "stages": stages,
        "seeds": arguments.seeds,
        "search_strategy": arguments.search_strategy,
        "search_seed": arguments.search_seed,
        "grid_size": len(grid),
        "max_trials": arguments.max_trials,
        "candidates_per_seed": len(candidates),
        "maximum_training_runs": sum(run.describe()["maximum_training_runs"] for run in runs),
        "cpu_threads": arguments.cpu_threads,
        "synthetic_smoke": arguments.smoke,
        "experiments": [run.describe() for run in runs],
    }
    return runs, stages, preview


def run_tuning_arguments(arguments, root):
    """Execute the resolved CLI stages with deferred testing and append exact invocation provenance."""
    import torch

    from .data import make_fixture

    runs, stages, preview = resolve_tuning_arguments(arguments, root)
    print(json.dumps(preview, indent=2), flush=True)
    if arguments.dry_run:
        return
    previous_threads = torch.get_num_threads()
    try:
        if arguments.cpu_threads is not None or arguments.smoke:
            torch.set_num_threads(arguments.cpu_threads or 2)
        if arguments.smoke:
            data_root = runs[0].data_root
            manifest = data_root / "manifests" / f"{arguments.dataset}.json"
            if not manifest.exists():
                make_fixture(data_root, arguments.dataset)
        with display_options(options_from_arguments(arguments)):
            for run in runs:
                with run:
                    invocation = {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "argv": sys.argv,
                        "arguments": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(arguments).items()
                        },
                        "seed": run.config.seed,
                        "stages": stages,
                        "cpu_threads": torch.get_num_threads(),
                    }
                    with (run.directory / "invocations.jsonl").open("a") as stream:
                        stream.write(json.dumps(invocation, allow_nan=False) + "\n")
                    run.run(stages=stages)
                print(f"tune results: {run.directory / 'comparison.md'}", flush=True)
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)
