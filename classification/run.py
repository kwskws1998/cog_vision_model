"""Fail-fast Python driver for preparation and the complete Conda classification study."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Enforce Conda and stop the entire stage after any unsuccessful subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "check", "smoke"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/classification_v1")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Use conda run --prefix ./.conda python classification/run.py ...")
    if arguments.workers < 0:
        parser.error("workers must be nonnegative")
    executable = [sys.executable, str(ROOT / "classification/experiment.py")]
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    common = ["--data-root", str(arguments.data_root.resolve())]
    if arguments.stage == "prepare":
        commands = [["check"], ["prepare", "--download", *common]]
        commands += [
            ["data-smoke", "--dataset", name, *common]
            for name in ("cifar100", "tiny_imagenet", "dominoes")
        ]
    elif arguments.stage == "train":
        commands = [
            [
                "study",
                *common,
                "--runs-dir",
                str(arguments.runs_dir.resolve()),
                "--workers",
                str(arguments.workers),
            ]
        ]
    else:
        commands = [[arguments.stage]]
    for command in commands:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(executable + command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
