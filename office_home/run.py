"""Run fail-fast Office-Home preparation or GPU training inside the existing Conda env."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Prepare on CPU first; gate the 36-run matrix on successful CUDA checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "office_home/configs/officehome_resnet18.json"
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/officehome_manifest.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/officehome_resnet18")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    arguments = parser.parse_args()
    for name in ("config", "manifest", "runs_dir"):
        setattr(arguments, name, getattr(arguments, name).resolve())
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error(
            "Use the project Conda interpreter: conda run --prefix ./.conda python office_home/run.py ..."
        )
    if (
        not arguments.seeds
        or len(set(arguments.seeds)) != len(arguments.seeds)
        or any(not 0 <= seed < 2**63 for seed in arguments.seeds)
    ):
        parser.error("Seeds must be unique integers in [0, 2**63)")
    executable = [sys.executable, str(ROOT / "office_home/experiment.py")]
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    if arguments.stage == "prepare":
        commands = [
            ["check"],
            ["prepare", "--download", "--manifest", str(arguments.manifest)],
            [
                "data-smoke",
                "--manifest",
                str(arguments.manifest),
                "--config",
                str(arguments.config),
            ],
        ]
    else:
        if not arguments.manifest.is_file():
            parser.error("Prepare Office-Home first with office_home/run.py prepare")
        commands = [
            ["profile", "--config", str(arguments.config)],
            [
                "data-smoke",
                "--device",
                "cuda",
                "--target",
                "Art",
                "--seed",
                str(min(arguments.seeds)),
                "--steps",
                "10",
                "--config",
                str(arguments.config),
                "--manifest",
                str(arguments.manifest),
            ],
            [
                "matrix",
                "--config",
                str(arguments.config),
                "--manifest",
                str(arguments.manifest),
                "--runs-dir",
                str(arguments.runs_dir),
                "--seeds",
                *map(str, arguments.seeds),
                "--evaluate",
            ],
        ]
    for command in commands:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(executable + command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
