"""Run the pinned ECCV 2022 CIFAR-100 CE baseline with full training data and explicit test metrics."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]


def build_parser():
    """Keep scientific settings in a JSON preset with ordinary command-line overrides."""
    from clvision.display import add_display_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "classification/configs/eccv2022_cifar100_baseline.json",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic CPU test only; never evidence of CIFAR accuracy",
    )
    for name in ("epochs", "batch-size", "eval-batch-size", "workers"):
        parser.add_argument("--" + name, type=int)
    for name in ("learning-rate", "momentum", "weight-decay", "lr-decay"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--milestones", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--schedule",
        choices=("official-code", "paper"),
        help="official-code drops at epochs 102/152; paper after epochs 100/150",
    )
    add_display_arguments(parser)
    return parser


def main(argv=None):
    """Validate and print the resolved protocol before performing an explicitly scoped run."""
    import torch
    from clvision.baseline_reference import BaselineConfig, checked_reference, load_data
    from clvision.baseline_training import train_seed
    from clvision.display import display_options, options_from_arguments, write
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor

    from cue_routing.io import atomic_json, read_json, run_lock

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raw = read_json(args.config)
        for key in BaselineConfig.__dataclass_fields__:
            value = getattr(args, key, None)
            if value is not None:
                raw[key] = value
        config = BaselineConfig(**raw).validate()
        options = options_from_arguments(args)
        checked_reference()
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.smoke:
        config.epochs, config.batch_size, config.eval_batch_size = 2, 4, 4
        config.workers, config.device, config.milestones = 0, "cpu", [1]
        config.seeds = [0]
        torch.set_num_threads(2)
    preview = config.preview() | {"synthetic_smoke": args.smoke}
    if args.smoke:
        preview.update(
            training_examples=8,
            evaluation_examples=4,
            evaluation_split="synthetic_test",
            published_cifar100_reference_percent=None,
        )
    print(json.dumps(preview, indent=2), flush=True)
    if args.dry_run:
        return
    if args.runs_dir is None:
        parser.error("--runs-dir is required; choose a fresh directory, or --resume the same one")
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Activate the project Conda environment before running this experiment")
    if config.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no automatic CPU fallback is permitted")
    if config.device == "cpu" and not args.smoke:
        torch.set_num_threads(min(8, torch.get_num_threads()))
    directory = args.runs_dir.resolve()
    with run_lock(directory), display_options(options):
        plan = {"config": asdict(config), "synthetic_smoke": args.smoke}
        plan_path = directory / "run_settings.json"
        if args.resume:
            if not plan_path.exists() or read_json(plan_path) != plan:
                parser.error(
                    "Resume requires the original run_settings.json and unchanged settings"
                )
        elif plan_path.exists():
            parser.error(
                "This directory already contains an experiment; use --resume or a new name"
            )
        if args.smoke:
            train = FakeData(
                size=8,
                image_size=(3, 32, 32),
                num_classes=100,
                transform=ToTensor(),
                random_offset=0,
            )
            test = FakeData(
                size=4,
                image_size=(3, 32, 32),
                num_classes=100,
                transform=ToTensor(),
                random_offset=100,
            )
            identity = {"kind": "synthetic_smoke_not_benchmark", "train_n": 8, "test_n": 4}
        else:
            train, test, identity = load_data(args.data_root, args.download)
        if not args.resume:
            atomic_json(plan_path, plan)
        summaries = []
        for index, seed in enumerate(config.seeds):
            run = directory / f"seed{seed}"
            write(
                f"[study] run={index + 1}/{len(config.seeds)} seed={seed} "
                f"train_n={len(train)} test_n={len(test)} remaining~=estimating"
            )
            summary = train_seed(
                config,
                seed,
                train,
                test,
                identity,
                run,
                resume=args.resume and (run / "contract.json").exists(),
            )
            summaries.append(summary)
            atomic_json(directory / "results.json", {"protocol": preview, "runs": summaries})
        lines = [
            "# ECCV 2022 author-code CE reference",
            "",
            "Synthetic smoke; not benchmark evidence."
            if args.smoke
            else "Full official CIFAR-100 training set; official test evaluated every epoch.",
            "",
            "| Seed | Final test accuracy | Best observed test accuracy | Best epoch |",
            "|---|---:|---:|---:|",
        ]
        for item in summaries:
            best = item["best_observed_test"]
            lines.append(
                f"| {item['seed']} | {100 * item['final_test']['accuracy']:.2f}% | "
                f"{100 * best['accuracy']:.2f}% | {best['epoch']} |"
            )
        lines += [
            "",
            "Best observed test accuracy is a maximum over test evaluations, not a "
            "validation-selected estimate. Final test uses the fixed final epoch.",
            "",
            "Published reference: 75.51 +/- 0.28% over five runs. "
            "A single seed does not reproduce that statistic.",
        ]
        (directory / "results.md").write_text("\n".join(lines) + "\n")
        write(f"Results: {directory / 'results.md'}")


if __name__ == "__main__":
    main()
