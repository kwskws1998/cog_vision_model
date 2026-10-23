"""Run one CIFAR-100 PreAct ERM baseline and time-bounded validation-only CUE tuning."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]


def main(argv=None):
    """Expose a single seed and explicit KST deadline without creating a hidden development seed."""
    import torch
    from clvision.data import make_fixture, prepare_cifar
    from clvision.display import add_display_arguments, display_options, options_from_arguments
    from clvision.overnight_settings import preview, validate_settings
    from clvision.overnight_workflow import run_overnight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "classification/configs/overnight_cifar100.json"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/classification")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--finish-by",
        required=True,
        help="ISO timestamp with timezone, e.g. 2026-10-24T10:30:00+09:00",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-vram-gib", type=float)
    parser.add_argument("--reserve-minutes", type=float)
    parser.add_argument("--time-safety-factor", type=float)
    parser.add_argument("--tau-candidates", type=float, nargs="+")
    parser.add_argument("--joint-floor-candidates", type=float, nargs="+")
    parser.add_argument("--gate-temperature", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Synthetic CPU verification; not benchmark accuracy"
    )
    add_display_arguments(parser)
    args = parser.parse_args(argv)
    try:
        settings = json.loads(args.config.read_text())
        for key in ("seed", "workers", "max_vram_gib"):
            if getattr(args, key) is not None:
                settings["training"][key] = getattr(args, key)
        for key in ("reserve_minutes", "time_safety_factor"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        if args.tau_candidates is not None or args.joint_floor_candidates is not None:
            taus = args.tau_candidates or list(
                dict.fromkeys(c["gate_tau"] for c in settings["candidates"])
            )
            floors = args.joint_floor_candidates or list(
                dict.fromkeys(c["joint_floor"] for c in settings["candidates"])
            )
            settings["candidates"] = [
                {"gate_tau": tau, "joint_floor": floor, "gate_temperature": 0.05}
                for floor in floors
                for tau in taus
            ]
        if args.gate_temperature is not None:
            for row in settings["candidates"]:
                row["gate_temperature"] = args.gate_temperature
        if args.smoke:
            settings["training"].update(
                debug=True,
                device="cpu",
                epochs=2,
                milestones=[1],
                workers=0,
                batch_size=2,
                eval_batch_size=4,
                debug_max_batches=1,
            )
            settings["candidates"] = settings["candidates"][:2]
            settings["reserve_minutes"] = 0
            torch.set_num_threads(2)
        validate_settings(settings)
        options = options_from_arguments(args)
        print(json.dumps(preview(settings, args.finish_by), indent=2), flush=True)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        return
    if not (Path(sys.prefix) / "conda-meta/history").is_file():
        parser.error("Activate the project Conda environment")
    if args.smoke:
        data = args.runs_dir / "synthetic_data"
        if not (data / "manifests/cifar100.json").exists():
            make_fixture(data, "cifar100")
    else:
        data = args.data_root
        if not (data / "manifests/cifar100.json").exists():
            prepare_cifar(data, download_data=args.download)
    with display_options(options):
        run_overnight(ROOT, data, args.runs_dir, settings, args.finish_by, args.resume)


if __name__ == "__main__":
    main()
