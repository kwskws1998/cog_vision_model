"""Run configurable, fail-fast Office-Home preparation or GPU comparison workflows."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "office_home"))


def build_parser():
    """Expose workflow settings and matching per-run hyperparameters without hidden defaults."""
    from ohvision.config import ExperimentConfig

    from cue_routing.arguments import add_config_arguments

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "train"):
        sub = stages.add_parser(stage, allow_abbrev=False)
        sub.add_argument(
            "--config", type=Path, default=ROOT / "office_home/configs/officehome_resnet18.json"
        )
        sub.add_argument("--manifest", type=Path, default=ROOT / "data/officehome_manifest.json")
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--smoke-steps", type=int)
        sub.add_argument("--dry-run", action="store_true", help="Validate and print commands only")
        add_config_arguments(sub, ExperimentConfig, exclude=("method", "target", "seed"))
        if stage == "train":
            sub.add_argument("--runs-dir", type=Path, default=ROOT / "runs/officehome_resnet18")
            sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            sub.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_commands(arguments):
    """Validate the resolved recipe and forward each explicit override to every GPU stage."""
    from ohvision.config import ExperimentConfig

    from cue_routing.arguments import (
        config_argument_tokens,
        load_config_arguments,
        load_data_smoke_config,
    )

    if arguments.threads < 1:
        raise ValueError("--threads must be positive")
    if arguments.smoke_steps is not None and arguments.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    overrides = config_argument_tokens(arguments, ExperimentConfig)
    smoke_overrides = config_argument_tokens(arguments, ExperimentConfig, exclude=("steps",))
    if arguments.steps is not None:
        smoke_overrides.extend(["--training-steps", str(arguments.steps)])
    config_path = ["--config", str(arguments.config)]
    manifest_path = ["--manifest", str(arguments.manifest)]
    if arguments.stage == "prepare":
        config, smoke_steps = load_data_smoke_config(arguments, ExperimentConfig)
        commands = [
            ["check"],
            ["prepare", "--download", *manifest_path],
            [
                "data-smoke",
                *manifest_path,
                *config_path,
                *smoke_overrides,
                "--steps",
                str(smoke_steps),
            ],
        ]
    else:
        config = load_config_arguments(arguments, ExperimentConfig)
        seeds = arguments.seeds
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(not 0 <= seed < 2**63 for seed in seeds)
        ):
            raise ValueError("Seeds must be unique integers in [0, 2**63)")
        if config.debug:
            raise ValueError("The benchmark matrix requires debug=false")
        if config.device != "cuda":
            raise ValueError("The train workflow requires --device cuda for its GPU checks")
        smoke_steps = arguments.smoke_steps if arguments.smoke_steps is not None else 10
        commands = [
            ["profile", *config_path, *overrides, "--seed", str(min(seeds))],
            [
                "data-smoke",
                *config_path,
                *smoke_overrides,
                *manifest_path,
                "--device",
                "cuda",
                "--target",
                "Art",
                "--seed",
                str(min(seeds)),
                "--steps",
                str(smoke_steps),
            ],
            [
                "matrix",
                *config_path,
                *overrides,
                *manifest_path,
                "--runs-dir",
                str(arguments.runs_dir),
                "--seeds",
                *map(str, seeds),
                *(["--evaluate"] if arguments.evaluate else []),
            ],
        ]
    return config, [["--threads", str(arguments.threads), *command] for command in commands]


def main(argv=None):
    """Print a reviewable plan or execute every stage with the selected Conda interpreter."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    for name in ("config", "manifest", "runs_dir"):
        if hasattr(arguments, name):
            setattr(arguments, name, getattr(arguments, name).resolve())
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error(
            "Use a Conda interpreter: conda activate cog-vision; python office_home/run.py ..."
        )
    try:
        config, commands = build_commands(arguments)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    executable = [sys.executable, str(ROOT / "office_home/experiment.py")]
    if arguments.dry_run:
        print(json.dumps({"resolved_config": config.to_dict(), "commands": commands}, indent=2))
        return
    if arguments.stage == "train" and not arguments.manifest.is_file():
        parser.error("Prepare Office-Home first with office_home/run.py prepare")
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for command in commands:
        print("Running: " + shlex.join(executable + command), flush=True)
        subprocess.run(executable + command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
