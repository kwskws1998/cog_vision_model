"""Edit inputs below; reuse the same training modules and reorder STAGES to change execution."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]

DATASETS = ("cifar100",)
SEEDS = (0,)
COMMON = dict(
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
    gate_tau=0.15,
    gate_temperature=0.05,
    joint_floor=0.5,
    gradient_clip=0.0,
    precision="fp32",
    eval_precision="fp32",
    device="cuda",
    workers=4,
    max_vram_gib=28.0,
)
DATASET_OVERRIDES = {"tiny_imagenet": {"normalization": "dataset"}}
TUNING = dict(enabled=True, strategy="grid", max_trials=6, sampling_seed=2027, metric="accuracy")
CUE_GRID = dict(joint_floor=[0.5, 0.0], gate_tau=[0.15, 0.05, 0.30], gate_temperature=[0.05])
FIXED_CUE = dict(gate_tau=0.15, joint_floor=0.5, gate_temperature=0.05)
BUDGET = dict(finish_by="2026-10-24T10:30:00+09:00", reserve_minutes=20, time_safety_factor=1.25)
PREFLIGHT_STEPS = 10
STAGES = ("preflight", "tune", "baseline", "evaluate")
DISPLAY = dict(progress="bar", logging_steps=200, progress_interval=1.0)


def candidates():
    """Choose a finite ordered grid or seeded random subset before observing any results."""
    from clvision.experiment_settings import cue_grid

    if type(TUNING["enabled"]) is not bool:
        raise ValueError("TUNING.enabled must be boolean")
    if not TUNING["enabled"]:
        return [dict(FIXED_CUE)]
    if TUNING["strategy"] not in ("grid", "random"):
        raise ValueError("TUNING.strategy must be grid or random")
    limit = TUNING["max_trials"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_trials must be a positive integer or None")
    rows = cue_grid(**CUE_GRID)
    if TUNING["strategy"] == "random":
        random.Random(TUNING["sampling_seed"]).shuffle(rows)
    return rows if limit is None else rows[:limit]


def main(argv=None):
    """Validate every job before execution, snapshot Python inputs, and run the explicit stages."""
    import torch
    from clvision.config import Config
    from clvision.data import make_fixture, prepare_cifar, prepare_tiny
    from clvision.display import DisplayOptions, display_options
    from clvision.experiment import Comparison
    from clvision.experiment_settings import Budget, Search, validate_stages

    from cue_routing.io import atomic_json, canonical_hash, read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.runs_dir is None:
        parser.error("--resume requires the original --runs-dir")
    if (
        not SEEDS
        or len(set(SEEDS)) != len(SEEDS)
        or not DATASETS
        or len(set(DATASETS)) != len(DATASETS)
    ):
        parser.error("SEEDS and DATASETS must be nonempty without duplicates")
    stages = validate_stages(STAGES)
    output = (
        args.runs_dir or ROOT / "runs" / datetime.now().strftime("erm_cue_%Y%m%d_%H%M%S_%f")
    ).resolve()
    search = Search(candidates(), metric=TUNING["metric"])
    budget = Budget() if args.smoke else Budget(**BUDGET)
    data_root = output / "synthetic_data" if args.smoke else args.data_root.resolve()
    jobs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            settings = {**COMMON, **DATASET_OVERRIDES.get(dataset, {})}
            if args.smoke:
                settings.update(
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
            jobs.append(
                Comparison(
                    root=ROOT,
                    data_root=data_root,
                    directory=output / dataset / f"seed{seed}",
                    config=Config(dataset=dataset, method="erm", seed=seed, **settings),
                    search=search,
                    budget=budget,
                    resume=args.resume,
                    preflight_steps=PREFLIGHT_STEPS,
                )
            )
    print(
        json.dumps(
            {"runs_dir": str(output), "stages": stages, "jobs": [job.describe() for job in jobs]},
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Activate the project Conda environment")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            "Output directory exists; choose a fresh --runs-dir or explicit --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    launch_plan = {
        "stages": list(stages),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "jobs": [job.describe() for job in jobs],
    }
    launch_path = output / "experiment_plan.json"
    if launch_path.exists() and canonical_hash(read_json(launch_path)) != canonical_hash(
        launch_plan
    ):
        raise ValueError("Resolved experiment inputs changed; use a new runs directory")
    atomic_json(launch_path, launch_plan)
    snapshot = output / "experiment.py"
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    if args.smoke:
        torch.set_num_threads(2)
    for dataset in DATASETS:
        if (data_root / "manifests" / f"{dataset}.json").exists():
            continue
        if args.smoke:
            make_fixture(data_root, dataset)
        else:
            prepare = prepare_cifar if dataset == "cifar100" else prepare_tiny
            prepare(data_root, download_data=args.download)
    with display_options(DisplayOptions(**DISPLAY)):
        for job in jobs:
            with job:
                job.run(stages=stages)


if __name__ == "__main__":
    main()
