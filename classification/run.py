"""Fail-fast Python driver for preparation and the complete Conda classification study."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]


def build_parser():
    """Share the exact same study inputs with the underlying experiment command."""
    from clvision.ablation_settings import add_ablation_arguments
    from clvision.arguments import add_study_arguments
    from clvision.config import ACTIVATIONS
    from clvision.cue_settings import add_cue_arguments, add_erm_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    add_cue_arguments(stages.add_parser("cue"), ROOT)
    add_erm_arguments(stages.add_parser("erm"), ROOT)
    add_ablation_arguments(stages.add_parser("ablation"), ROOT)
    for name in ("check", "smoke"):
        stage = stages.add_parser(name)
        if name == "smoke":
            stage.add_argument("--activation", choices=ACTIVATIONS, default="postact")
    prepare = stages.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    for name in ("train", "plan"):
        stage = stages.add_parser(name)
        stage.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
        stage.add_argument("--runs-dir", type=Path, default=ROOT / "runs/classification_2seeds_v1")
        add_study_arguments(stage, ROOT)
        if name == "train":
            stage.add_argument("--dry-run", action="store_true")
    return parser


def main():
    """Enforce Conda and stop the entire stage after any unsuccessful subprocess."""
    from clvision.arguments import load_study_arguments, study_argument_tokens, study_preview
    from clvision.display import options_from_arguments

    parser = build_parser()
    arguments = parser.parse_args()
    try:
        options_from_arguments(arguments)
    except ValueError as error:
        parser.error(str(error))
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Use conda run --prefix ./.conda python classification/run.py ...")
    if arguments.stage in ("cue", "erm", "ablation"):
        from clvision.ablation_settings import load_ablation_arguments
        from clvision.cue_settings import CueSettings, ErmSettings, load_cue_arguments

        try:
            settings = (
                load_ablation_arguments(arguments)
                if arguments.stage == "ablation"
                else load_cue_arguments(
                    arguments, ErmSettings if arguments.stage == "erm" else CueSettings
                )
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        if arguments.dry_run:
            print(json.dumps(settings.preview(), indent=2))
            return
    if arguments.stage in ("train", "plan"):
        try:
            settings = load_study_arguments(arguments)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        if arguments.stage == "plan" or arguments.dry_run:
            print(json.dumps(study_preview(settings), indent=2))
            return
    executable = [sys.executable, "-u", str(ROOT / "classification/experiment.py")]
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    common = (
        ["--data-root", str(arguments.data_root.resolve())]
        if hasattr(arguments, "data_root")
        else []
    )
    if arguments.stage in ("cue", "erm", "ablation"):
        commands = [[arguments.stage, *sys.argv[2:]]]
    elif arguments.stage == "prepare":
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
                *study_argument_tokens(arguments),
            ]
        ]
    elif arguments.stage == "smoke":
        commands = [["smoke", "--activation", arguments.activation]]
    else:
        commands = [[arguments.stage]]
    for command in commands:
        print("Running: " + shlex.join(executable + command), flush=True)
        subprocess.run(executable + command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
