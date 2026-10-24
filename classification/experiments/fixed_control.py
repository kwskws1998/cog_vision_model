"""Editable single-control recipe; preflight, train, then evaluate the sealed final test."""

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
    method="fixed_global_local",
    seed=0,
    fixed_lambda=0.5,
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
REFERENCE = dict(
    expected_dataset_id="c2bf1ba3ae3dd4d810b6884b8a2f8964efc495398eadb1205edb82e343113d7f",
    expected_initial_weights="14390f2005fb56f90d10372941fda4a2a374867a56e579f28fdde1ca9ab2cb82",
)
BUDGET = dict(finish_by=None, reserve_minutes=10, time_safety_factor=1.25)
PREFLIGHT_STEPS = 10
DISPLAY = dict(progress="bar", logging_steps=200, progress_interval=1.0)


def main(argv=None):
    """Resolve explicit inputs, support isolated smoke/dry runs, and call reusable stages."""
    import torch
    from clvision.config import Config
    from clvision.data import make_fixture
    from clvision.display import DisplayOptions, display_options
    from clvision.experiment_settings import Budget
    from clvision.single_experiment import SingleRun

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
        args.runs_dir or ROOT / "runs" / datetime.now().strftime("fixed_control_%Y%m%d_%H%M%S_%f")
    ).resolve()
    recipe, reference = dict(RECIPE), dict(REFERENCE)
    data_root = args.data_root.resolve()
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
        reference = {}
        data_root = directory / "synthetic_data"
    run = SingleRun(
        root=ROOT,
        data_root=data_root,
        directory=directory,
        config=Config(**recipe),
        budget=Budget() if args.smoke else Budget(**BUDGET),
        resume=args.resume,
        preflight_steps=PREFLIGHT_STEPS,
        experiment_source=Path(__file__),
        **reference,
    )
    print(json.dumps({"runs_dir": str(directory), **run.describe()}, indent=2), flush=True)
    if args.dry_run:
        return
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Activate the project Conda environment")
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError("Output directory exists; use a new directory or --resume")
    if args.smoke:
        torch.set_num_threads(2)
        if not (data_root / "manifests" / f"{recipe['dataset']}.json").exists():
            make_fixture(data_root, recipe["dataset"])
    with display_options(DisplayOptions(**DISPLAY)), run:
        run.preflight()
        run.train()
        run.evaluate()


if __name__ == "__main__":
    main()
