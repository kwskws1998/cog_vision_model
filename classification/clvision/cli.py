"""Conda-only preparation, verification, preflight, and complete registered experiment commands."""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from cue_routing.io import atomic_json, environment, file_hash, read_json

from .arguments import add_study_arguments, load_study_arguments, study_preview
from .config import MAIN_METHODS, recipe
from .cue_settings import add_cue_arguments, load_cue_arguments
from .cue_workflow import run_cue_study
from .data import (
    load_manifest,
    make_fixture,
    make_loader,
    prepare_cifar,
    prepare_corruptions,
    prepare_dominoes,
    prepare_tiny,
)
from .engine import make_model, optimize_batch, optimizer_for
from .identity import source_hash
from .profile import profile_dataset
from .workflow import base_recipe, run_study


def require_conda():
    """Avoid silently using a global or virtualenv interpreter on the training server."""
    if not (Path(sys.prefix) / "conda-meta" / "history").is_file():
        raise RuntimeError(
            "Use conda run --prefix ./.conda python classification/experiment.py ..."
        )


def check(root):
    """Run legacy and new tests together and record actual outcomes against current sources."""
    from ohvision.fingerprint import source_hash as officehome_source_hash

    suite = unittest.TestSuite(
        [
            unittest.TestLoader().discover(str(root / directory))
            for directory in ("tests", "office_home/tests", "classification/tests")
        ]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "source_sha256": source_hash(root),
        "officehome_source_sha256": officehome_source_hash(root),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "environment": environment(),
    }
    atomic_json(root / "classification/verification/unit_tests.json", report)
    if not report["passed"]:
        raise SystemExit(1)
    return report


def prepare(root, data_root, datasets, download):
    """Prepare selected datasets with a fixed root and no GPU training or final predictions."""
    functions = {
        "cifar100": prepare_cifar,
        "tiny_imagenet": prepare_tiny,
        "dominoes": prepare_dominoes,
        "cifar100_c": prepare_corruptions,
    }
    results = {}
    for name in datasets:
        print(f"Preparing {name}", flush=True)
        value = functions[name](data_root, download_data=download)
        verified = load_manifest(Path(data_root) / "manifests" / f"{name}.json")
        results[name] = {
            "dataset_id": value["dataset_id"],
            "classes": len(value.get("classes", [])),
            "counts": {key: len(indices) for key, indices in value.get("splits", {}).items()},
            "verified": verified["dataset_id"] == value["dataset_id"],
        }
    report = {
        "source_sha256": source_hash(root),
        "datasets": results,
        "target_predictions_computed": False,
    }
    atomic_json(root / "classification/verification/dataset_preparation.json", report)
    return report


def data_smoke(root, manifest, device="cpu", steps=1):
    """Verify matched real-source updates without constructing any final-test loader."""
    if steps < 1:
        raise ValueError("At least one smoke update is required")
    result = {
        "source_sha256": source_hash(root),
        "dataset_id": manifest["dataset_id"],
        "benchmark_evidence": False,
        "target_loader_accessed": False,
        "methods": {},
        "device": device,
    }
    for method in MAIN_METHODS:
        config = recipe(
            manifest["dataset"],
            method=method,
            debug=True,
            device=device,
            precision="fp32" if device == "cpu" else "fp16",
            batch_size=2 if device == "cpu" else 128,
            workers=0,
        )
        from cue_routing.io import seed_everything

        seed_everything(0, device)
        model = make_model(config)
        initial = model.shared_state_hash()
        before = next(model.parameters()).detach().clone()
        optimizer = optimizer_for(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        loader = make_loader(manifest, config, "train", epoch=1)
        batch = next(iter(loader))
        for _ in range(steps):
            diagnostics = optimize_batch(
                model, optimizer, scaler, batch[0].to(device), batch[1].to(device), config
            )
        result["methods"][method] = {
            "initial_shared_model_sha256": initial,
            "first_augmented_batch_sha256": hashlib.sha256(
                batch[0].numpy().tobytes() + batch[1].numpy().tobytes()
            ).hexdigest(),
            "optimizer_steps": steps,
            "backbone_updated": not torch.equal(before, next(model.parameters()).detach()),
            "diagnostics": diagnostics,
        }
    result["matched_initial_weights"] = (
        len({item["initial_shared_model_sha256"] for item in result["methods"].values()}) == 1
    )
    result["matched_augmented_batches"] = (
        len({item["first_augmented_batch_sha256"] for item in result["methods"].values()}) == 1
    )
    if (
        not result["matched_initial_weights"]
        or not result["matched_augmented_batches"]
        or not all(item["backbone_updated"] for item in result["methods"].values())
    ):
        raise AssertionError("Smoke did not update matched backbones")
    atomic_json(
        root / "classification/verification" / f"{manifest['dataset']}_{device}_smoke.json", result
    )
    return result


def smoke(root):
    """Exercise every stage of a complete reduced study using explicit synthetic fixtures."""
    with tempfile.TemporaryDirectory(prefix="classification-smoke-") as temporary:
        temporary = Path(temporary)
        for dataset in ("cifar100", "tiny_imagenet", "dominoes"):
            make_fixture(temporary / "data", dataset)
        result = run_study(root, temporary / "data", temporary / "study", debug=True, workers=0)
        progress = read_json(temporary / "study/progress.json")
        if (
            progress["status"] != "complete"
            or progress["completed_training_runs"] != progress["total_training_runs"]
            or progress["completed_clean_evaluations"] != progress["total_clean_evaluations"]
            or progress["estimated_remaining_seconds"] != 0
        ):
            raise AssertionError("Complete study progress does not match the executed scope")
        completions = {
            str(path): file_hash(path) for path in (temporary / "study").rglob("completion.json")
        }
        repeated = run_study(root, temporary / "data", temporary / "study", debug=True, workers=0)
        resumed_progress = read_json(temporary / "study/progress.json")
        if repeated != result or any(
            file_hash(path) != digest for path, digest in completions.items()
        ):
            raise AssertionError("A completed study was not reused exactly")
        if resumed_progress["completed_training_runs"] != progress["completed_training_runs"]:
            raise AssertionError("Resumed progress lost completed runs")
        prediction_path = temporary / "study/main/cifar100/seed0_erm/test_predictions.pt"
        metrics_path = prediction_path.with_name("test_metrics.json")
        saved_predictions, saved_metrics = prediction_path.read_bytes(), metrics_path.read_bytes()
        values = torch.load(prediction_path, weights_only=True)
        values["sample_ids"] = list(reversed(values["sample_ids"]))
        torch.save(values, prediction_path)
        metrics = read_json(metrics_path)
        metrics["prediction_sha256"] = file_hash(prediction_path)
        atomic_json(metrics_path, metrics)
        try:
            try:
                run_study(root, temporary / "data", temporary / "study", debug=True, workers=0)
            except ValueError as error:
                if "order" not in str(error):
                    raise
            else:
                raise AssertionError("Reordered final predictions passed the report guard")
        finally:
            prediction_path.write_bytes(saved_predictions)
            metrics_path.write_bytes(saved_metrics)
    report = {
        "source_sha256": source_hash(root),
        "passed": True,
        "benchmark_evidence": False,
        "datasets": list(result["datasets"]),
        "purpose": "synthetic_complete_study_verification",
        "completed_study_reused": True,
        "reordered_predictions_rejected": True,
        "whole_study_progress_verified": True,
        "resumed_progress_verified": True,
    }
    atomic_json(root / "classification/verification/cpu_smoke.json", report)
    return report


def main(root):
    """Parse explicit stages and fail before benchmark creation when preflight is unavailable."""
    require_conda()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_cue_arguments(commands.add_parser("cue"), root)
    for name in ("check", "smoke"):
        commands.add_parser(name)
    for name in ("prepare", "data-smoke", "profile", "study", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", type=Path, default=root / "data/classification")
        if name in ("profile", "study", "plan"):
            add_study_arguments(command, root)
        if name in ("study", "plan"):
            command.add_argument(
                "--runs-dir", type=Path, default=root / "runs/classification_2seeds_v1"
            )
            if name == "study":
                command.add_argument("--dry-run", action="store_true")
        if name in ("prepare", "data-smoke", "profile"):
            command.add_argument(
                "--dataset",
                choices=("all", "cifar100", "tiny_imagenet", "dominoes", "cifar100_c"),
                default="all",
            )
        if name == "prepare":
            command.add_argument("--download", action="store_true")
        if name == "data-smoke":
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--steps", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.command == "cue":
        try:
            settings = load_cue_arguments(arguments)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(settings.preview(), indent=2), flush=True)
        if arguments.dry_run:
            return
        torch.set_num_threads(4)
        run_cue_study(root, arguments.data_root, arguments.runs_dir, settings)
        print(f"CUE results: {arguments.runs_dir.resolve() / 'comparison.md'}", flush=True)
        return
    settings = None
    if arguments.command in ("profile", "study", "plan"):
        try:
            settings = load_study_arguments(arguments)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if arguments.command == "plan" or getattr(arguments, "dry_run", False):
        print(json.dumps(study_preview(settings), indent=2))
        return
    torch.set_num_threads(4)
    if arguments.command == "check":
        value = check(root)
    elif arguments.command == "prepare":
        datasets = (
            ("cifar100", "tiny_imagenet", "dominoes", "cifar100_c")
            if arguments.dataset == "all"
            else (arguments.dataset,)
        )
        value = prepare(root, arguments.data_root, datasets, arguments.download)
    elif arguments.command == "smoke":
        value = smoke(root)
    elif arguments.command == "study":
        print(json.dumps(study_preview(settings), indent=2), flush=True)
        result = run_study(root, arguments.data_root, arguments.runs_dir, settings=settings)
        value = {
            "status": "complete",
            "benchmark_evidence": result["benchmark_evidence"],
            "comparison_json": str(arguments.runs_dir.resolve() / "comparison.json"),
            "comparison_markdown": str(arguments.runs_dir.resolve() / "comparison.md"),
            "rankings": {name: item["ranking"] for name, item in result["datasets"].items()},
        }
    else:
        if arguments.dataset in ("all", "cifar100_c"):
            parser.error("Choose an explicit trainable dataset")
        manifest = load_manifest(
            arguments.data_root / "manifests" / f"{arguments.dataset}.json",
            root_override=arguments.data_root,
        )
        if arguments.command == "data-smoke":
            value = data_smoke(root, manifest, arguments.device, arguments.steps)
        else:
            base = base_recipe(arguments.dataset, settings)
            value = profile_dataset(
                root,
                manifest,
                base,
                root / "classification/verification" / f"{arguments.dataset}_cuda.json",
            )
            if not value["measured"]:
                print(json.dumps(value, indent=2))
                raise SystemExit(2)
            if not value["passes_budget"]:
                print(json.dumps(value, indent=2))
                raise SystemExit(1)
    print(json.dumps(value, indent=2))
