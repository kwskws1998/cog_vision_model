"""Edit these Python inputs to run additional CUE candidates without retraining baselines."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]

RECIPE = dict(
    dataset="cifar100",
    method="cue",
    seed=0,
    activation="preact",
    initialization="msra",
    normalization="zipfls",
    tiny_imagenet_size=64,
    epochs=200,
    batch_size=128,
    eval_batch_size=128,
    learning_rate=0.1,
    warmup_epochs=0,
    momentum=0.9,
    weight_decay=1e-4,
    milestones=(100, 150),
    lr_decay=0.1,
    gate_tau=0.30,
    gate_temperature=0.05,
    joint_floor=0.5,
    gradient_clip=0.0,
    precision="fp32",
    eval_precision="fp32",
    device="cuda",
    workers=4,
    max_vram_gib=28.0,
)
CANDIDATES = [
    dict(gate_tau=0.30, joint_floor=0.5, gate_temperature=0.025),
    dict(gate_tau=0.30, joint_floor=0.5, gate_temperature=0.10),
]
REFERENCE_FILE = ROOT / "classification/references/cifar100_seed0_cue_temperature005.json"
SELECTION_METRIC = "accuracy"
BUDGET = dict(finish_by=None, reserve_minutes=10, time_safety_factor=1.25)
PREFLIGHT_STEPS = 10
STAGES = ("preflight", "tune", "evaluate")
DISPLAY = dict(progress="bar", logging_steps=200, progress_interval=1.0)


def main(argv=None):
    """Snapshot inputs, execute only declared trials, and test at most one validation winner."""
    import torch
    from clvision.config import Config
    from clvision.data import make_fixture
    from clvision.display import DisplayOptions, display_options
    from clvision.experiment import Comparison
    from clvision.experiment_settings import Budget, Search, validate_stages

    from cue_routing.io import read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.runs_dir is None:
        parser.error("--resume requires the original --runs-dir")
    directory = (
        args.runs_dir or ROOT / "runs" / datetime.now().strftime("cue_temperature_%Y%m%d_%H%M%S_%f")
    ).resolve()
    recipe = dict(RECIPE)
    data_root = args.data_root.resolve()
    reference = None if REFERENCE_FILE is None else read_json(REFERENCE_FILE)
    if args.smoke:
        recipe.update(
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
        reference = None
        data_root = directory / "synthetic_data"
    stages = validate_stages(STAGES, include_baseline=False)
    run = Comparison(
        root=ROOT,
        data_root=data_root,
        directory=directory,
        config=Config(**recipe),
        search=Search(CANDIDATES, metric=SELECTION_METRIC),
        budget=Budget() if args.smoke else Budget(**BUDGET),
        resume=args.resume,
        preflight_steps=PREFLIGHT_STEPS,
        include_baseline=False,
        reference=reference,
        experiment_source=Path(__file__),
    )
    print(
        json.dumps({"runs_dir": str(directory), "stages": stages, **run.describe()}, indent=2),
        flush=True,
    )
    if args.dry_run:
        return
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Activate the project Conda environment")
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError("Output directory exists; use a fresh directory or --resume")
    if args.smoke:
        torch.set_num_threads(2)
        if not (data_root / "manifests" / f"{recipe['dataset']}.json").exists():
            make_fixture(data_root, recipe["dataset"])
    with display_options(DisplayOptions(**DISPLAY)), run:
        run.run(stages=stages)


if __name__ == "__main__":
    main()
