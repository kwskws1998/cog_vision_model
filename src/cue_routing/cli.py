"""Python-only experiment commands and a fully specified sequential benchmark matrix."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from .config import DOMAINS, METHODS, ExperimentConfig
from .data import (
    download_pacs,
    evaluation_loader,
    extract_archive,
    load_manifest,
    make_fixture,
    prepare_manifest,
    training_loader,
)
from .engine import (
    autocast_context,
    evaluate_target,
    make_model,
    optimize_batch,
    train,
    validate_device,
    vram_stats,
)
from .io import (
    atomic_json,
    canonical_hash,
    environment,
    file_hash,
    read_json,
    run_lock,
    seed_everything,
    source_hash,
    state_hash,
)
from .metrics import classification_metrics
from .profile import profile_memory
from .protocol import verify_selection


def load_config(arguments):
    """Apply only explicitly supplied method, target, and seed overrides."""
    config = ExperimentConfig.load(arguments.config)
    updates = {
        key: getattr(arguments, key)
        for key in ("method", "target", "seed")
        if getattr(arguments, key, None) is not None
    }
    return replace(config, **updates).validate()


def check(project_root):
    """Run all numerical, protocol, data, and checkpoint tests without downloading data."""
    suite = unittest.defaultTestLoader.discover(str(Path(project_root) / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "source_sha256": source_hash(project_root),
        "environment": environment(),
    }
    atomic_json(Path(project_root) / "verification" / "unit_tests.json", report)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return report


def smoke(project_root):
    """Exercise real ResNet-18 training and sealed evaluation on tiny artificial images."""
    root = Path(project_root)
    seed_everything(0)
    result = {
        "purpose": "synthetic_cpu_pipeline_verification",
        "benchmark_evidence": False,
        "environment": environment(),
        "methods": {},
    }
    with tempfile.TemporaryDirectory(prefix="cue-smoke-") as temporary:
        temporary = Path(temporary)
        manifest = make_fixture(temporary / "images", root / "references" / "pacs_splits")
        for method in METHODS:
            config = ExperimentConfig(
                method=method,
                steps=2,
                eval_every=1,
                batch_per_domain=1,
                image_size=64,
                eval_batch_size=7,
                device="cpu",
                precision="fp32",
                workers=0,
                pretrained=False,
                debug=True,
            )
            run_dir = temporary / method
            training = train(config, manifest, run_dir, root)
            evaluation = evaluate_target(manifest, run_dir, root)
            result["methods"][method] = {
                "training": training,
                "target_metric_count": evaluation["metrics"]["n"],
                "target_access_events": len(read_json(run_dir / "target_access.json")),
            }
        contracts = [read_json(temporary / method / "contract.json") for method in METHODS]
        result["matched_initial_weights"] = (
            len({item["initial_model_sha256"] for item in contracts}) == 1
        )
        if not result["matched_initial_weights"]:
            raise AssertionError("Arms did not start from identical weights")
    result["source_sha256"] = source_hash(root)
    atomic_json(root / "verification" / "cpu_smoke.json", result)
    return result


def pretrained_smoke(project_root):
    """Verify one pretrained 224px CPU optimizer step per arm without efficacy claims."""
    result = {
        "purpose": "pretrained_224px_synthetic_cpu_verification",
        "benchmark_evidence": False,
        "cuda_memory_measured": False,
        "methods": {},
    }
    initial_hashes = []
    for method in METHODS:
        config = ExperimentConfig(
            method=method,
            steps=1,
            eval_every=1,
            batch_per_domain=1,
            image_size=224,
            device="cpu",
            precision="fp32",
            workers=0,
            pretrained=True,
            debug=True,
        )
        seed_everything(config.seed)
        model = make_model(config, project_root).train()
        initial_hashes.append(state_hash(model.state_dict()))
        before = model.encoder[0].weight.detach().clone()
        generator = torch.Generator().manual_seed(812)
        images = torch.randn(3, 3, 224, 224, generator=generator)
        labels = torch.tensor([0, 2, 6])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        diagnostics = optimize_batch(model, optimizer, scaler, images, labels, config)
        changed = not torch.equal(before, model.encoder[0].weight)
        if not changed:
            raise AssertionError("Backbone did not receive an optimizer update")
        result["methods"][method] = {
            "diagnostics": diagnostics,
            "backbone_updated": changed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    result["matched_initial_weights"] = len(set(initial_hashes)) == 1
    if not result["matched_initial_weights"]:
        raise AssertionError("Pretrained initial weights differ across methods")
    result["initial_model_sha256"] = initial_hashes[0]
    result["environment"] = environment()
    result["source_sha256"] = source_hash(project_root)
    atomic_json(Path(project_root) / "verification" / "pretrained_cpu_smoke.json", result)
    return result


def real_data_smoke(manifest_path, project_root, config=None, steps=1):
    """Verify pretrained source updates on CPU or the unchanged CUDA benchmark recipe."""
    if type(steps) is not int or steps < 1:
        raise ValueError("data-smoke steps must be a positive integer")
    if config is None:
        config = ExperimentConfig(
            steps=1,
            eval_every=1,
            batch_per_domain=1,
            image_size=224,
            eval_batch_size=3,
            device="cpu",
            precision="fp32",
            workers=0,
            pretrained=True,
            debug=True,
        )
    validate_device(config)
    if not config.pretrained or config.image_size != 224:
        raise ValueError("data-smoke requires pretrained weights and 224px images")
    manifest = load_manifest(manifest_path, Path(project_root) / "references" / "pacs_splits")
    if manifest["kind"] != "pacs":
        raise ValueError("data-smoke requires the complete real PACS manifest")
    result = {
        "purpose": f"real_PACS_source_data_{config.device}_smoke",
        "benchmark_evidence": False,
        "cuda_memory_measured": config.device == "cuda",
        "config": config.to_dict(),
        "optimizer_steps_per_method": steps,
        "dataset_id": manifest["dataset_id"],
        "verified_image_count": len(manifest["images"]),
        "target_loader_accessed": False,
        "methods": {},
    }
    initial_hashes, batch_hashes = [], []
    for method in METHODS:
        current = replace(config, method=method)
        seed_everything(current.seed, current.device)
        if current.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = make_model(current, project_root).train()
        initial_hashes.append(state_hash(model.state_dict()))
        before = model.encoder[0].weight.detach().clone()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=current.learning_rate, weight_decay=current.weight_decay
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=current.device == "cuda" and current.precision == "fp16"
        )
        method_batches, step_diagnostics = [], []
        loader = training_loader(manifest, current, 0, steps)
        for images, labels in loader:
            method_batches.append(state_hash({"images": images, "labels": labels}))
            images = images.to(current.device, non_blocking=True)
            labels = labels.to(current.device, non_blocking=True)
            step_diagnostics.append(
                optimize_batch(model, optimizer, scaler, images, labels, current)
            )
            vram_stats(current)
        if len(step_diagnostics) != steps or any(
            state.get("step") is not None and int(state["step"]) != steps
            for state in optimizer.state.values()
        ):
            raise AssertionError("data-smoke did not complete every scheduled Adam update")
        batch_hashes.append(canonical_hash(method_batches))
        checked = []
        model.eval()
        for domain in DOMAINS:
            if domain != current.target:
                batch, _ = next(iter(evaluation_loader(manifest, domain, "crossval", current)))
                with torch.no_grad(), autocast_context(current):
                    output = model(batch.to(current.device, non_blocking=True))
                if output.shape != (len(batch), 7) or not torch.isfinite(output).all():
                    raise AssertionError("Invalid real-source validation predictions")
                checked.append(domain)
                vram_stats(current)
        updated = not torch.equal(before, model.encoder[0].weight)
        if not updated:
            raise AssertionError("Real-source optimizer step did not update the backbone")
        result["methods"][method] = {
            "diagnostics": step_diagnostics[-1],
            "step_diagnostics": step_diagnostics,
            "amp_overflow_retries_total": sum(
                item["amp_overflow_retries"] for item in step_diagnostics
            ),
            "optimizer_steps": steps,
            "backbone_updated": updated,
            "source_validation_domains_checked": checked,
            "training_batch": len(images),
            "image_size": 224,
            "vram": vram_stats(current),
        }
        del model, optimizer, scaler, before, images, labels, batch, output, loader
    result["matched_initial_weights"] = len(set(initial_hashes)) == 1
    result["matched_training_images_and_augmentations"] = len(set(batch_hashes)) == 1
    if (
        not result["matched_initial_weights"]
        or not result["matched_training_images_and_augmentations"]
    ):
        raise AssertionError("Real-source comparison is not matched")
    result["environment"] = environment()
    result["source_sha256"] = source_hash(project_root)
    atomic_json(
        Path(project_root) / "verification" / f"real_data_{config.device}_smoke.json", result
    )
    return result


def validate_matrix_runs(plan, directory, require_complete=False):
    """Check every existing arm and matched initial weights before any target access."""
    complete, initial_weights = set(), {}
    for specification in plan["runs"]:
        run_dir = (
            Path(directory)
            / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
        )
        if not (run_dir / "contract.json").exists():
            if require_complete or (run_dir / "target_access.json").exists():
                raise ValueError("Missing training contract in the matrix")
            continue
        contract = read_json(run_dir / "contract.json")
        if (
            contract["config"] != {**plan["base_config"], **specification}
            or contract["dataset_id"] != plan["dataset_id"]
            or contract["source_sha256"] != plan["source_sha256"]
            or contract["runtime"] != environment()
        ):
            raise ValueError(
                "Existing run differs from the matrix recipe, dataset, source, or runtime"
            )
        key = (specification["target"], specification["seed"])
        if key in initial_weights and initial_weights[key] != contract["initial_model_sha256"]:
            raise ValueError("Methods do not form a matched comparison of initial weights")
        initial_weights[key] = contract["initial_model_sha256"]
        if (run_dir / "selection.json").exists():
            verify_selection(run_dir, contract)
            complete.add(run_dir.name)
        elif require_complete or (run_dir / "target_access.json").exists():
            raise ValueError("An unsealed or incomplete run cannot enter matrix evaluation")
    return complete


def matrix(config, manifest, directory, root, seeds, evaluate=False):
    """Finish every source-only training run before any optional target evaluations."""
    directory = Path(directory)
    config.validate()
    if (
        config.debug
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("The benchmark matrix requires unique seeds and debug=false")
    plan = [
        {"method": method, "target": domain, "seed": seed}
        for seed in sorted(seeds)
        for domain in DOMAINS
        for method in METHODS
    ]
    data = load_manifest(manifest, Path(root) / "references" / "pacs_splits")
    if data["kind"] != "pacs":
        raise ValueError("Benchmark matrices require real PACS images")
    seed_everything(config.seed, config.device)
    plan_contract = {
        "schema": 2,
        "base_config": config.to_dict(),
        "runs": plan,
        "dataset_id": data["dataset_id"],
        "source_sha256": source_hash(root),
    }
    with run_lock(directory):
        if (directory / "matrix_plan.json").exists() and read_json(
            directory / "matrix_plan.json"
        ) != plan_contract:
            raise ValueError("Existing matrix plan differs; use another directory")
        complete = validate_matrix_runs(plan_contract, directory)
        atomic_json(directory / "matrix_plan.json", plan_contract)
        for specification in plan:
            current = replace(config, **specification).validate()
            run_dir = directory / f"{current.target}_seed{current.seed}_{current.method}"
            if run_dir.name not in complete:
                train(current, manifest, run_dir, root, resume=(run_dir / "contract.json").exists())
        validate_matrix_runs(plan_contract, directory, require_complete=True)
        if evaluate:
            for specification in plan:
                run_dir = (
                    directory
                    / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
                )
                if not (run_dir / "target_metrics.json").exists():
                    evaluate_target(manifest, run_dir, root)
            return report_matrix(directory)
        return {
            "completed_runs": len(plan),
            "reused_completed_runs": len(complete),
            "target_evaluation_performed": False,
        }


def report_matrix(directory):
    """Aggregate only complete matched matrices, with variation across seed averages."""
    directory = Path(directory)
    plan = read_json(directory / "matrix_plan.json")
    planned_seeds = sorted({item["seed"] for item in plan["runs"]})
    expected = [
        {"method": method, "target": domain, "seed": seed}
        for seed in planned_seeds
        for domain in DOMAINS
        for method in METHODS
    ]
    if not planned_seeds or plan["runs"] != expected:
        raise ValueError("Report requires a complete, unique four-domain, three-method matrix")
    records, controls = [], {}
    dataset_ids = set()
    for specification in plan["runs"]:
        run_dir = (
            directory
            / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
        )
        contract = read_json(run_dir / "contract.json")
        seal = verify_selection(run_dir, contract)
        result = read_json(run_dir / "target_metrics.json")
        expected_config = {**plan["base_config"], **specification}
        if (
            contract["config"] != expected_config
            or contract["source_sha256"] != plan["source_sha256"]
        ):
            raise ValueError("Run differs from the matrix's prespecified recipe or code")
        dataset_ids.add(contract["dataset_id"])
        if contract["dataset_id"] != plan["dataset_id"]:
            raise ValueError("Run dataset differs from the prespecified matrix")
        if (
            not result["benchmark_evidence"]
            or result["contract_hash"] != canonical_hash(contract)
            or result["selection"] != seal
            or any(result[key] != value for key, value in specification.items())
        ):
            raise ValueError("Target results lack valid benchmark provenance")
        if result["prediction_sha256"] != file_hash(run_dir / "target_predictions.pt"):
            raise ValueError("Target predictions changed after evaluation")
        predictions = torch.load(
            run_dir / "target_predictions.pt", map_location="cpu", weights_only=True
        )
        paths = predictions["paths"]
        partition = [[path, label] for path, label in zip(paths, predictions["targets"].tolist())]
        if (
            len(paths) != contract["target_n"]
            or len(paths) != len(set(paths))
            or len(paths) != len(predictions["targets"])
            or canonical_hash(partition) != contract["target_partition_sha256"]
        ):
            raise ValueError("Saved predictions do not match the exact held-out target partition")
        if (
            predictions["contract_hash"] != canonical_hash(contract)
            or classification_metrics(predictions["logits"], predictions["targets"])
            != result["metrics"]
        ):
            raise ValueError("Reported metrics do not match saved target predictions")
        config = {key: value for key, value in contract["config"].items() if key != "method"}
        control = {
            "config": config,
            "dataset_id": contract["dataset_id"],
            "source_sha256": contract["source_sha256"],
            "initial_model_sha256": contract["initial_model_sha256"],
            "runtime": contract["runtime"],
        }
        key = (result["target"], result["seed"])
        if key in controls and controls[key] != control:
            raise ValueError("Methods do not form a matched comparison")
        controls[key] = control
        records.append(result)
    if len(dataset_ids) != 1:
        raise ValueError("The matrix mixes different datasets")
    seeds = sorted({item["seed"] for item in records})
    summary = {}
    for method in METHODS:
        selected = [item for item in records if item["method"] == method]
        seed_means = {
            str(seed): statistics.mean(
                item["metrics"]["accuracy"] for item in selected if item["seed"] == seed
            )
            for seed in seeds
        }
        domain_means = {
            domain: statistics.mean(
                item["metrics"]["accuracy"] for item in selected if item["target"] == domain
            )
            for domain in DOMAINS
        }
        summary[method] = {
            "mean_accuracy": statistics.mean(seed_means.values()),
            "seed_sd": statistics.stdev(seed_means.values()) if len(seeds) > 1 else None,
            "seed_macro_domain_accuracies": seed_means,
            "domain_mean_accuracies": domain_means,
            "mean_nll": statistics.mean(item["metrics"]["nll"] for item in selected),
            "mean_ece": statistics.mean(item["metrics"]["ece"] for item in selected),
        }
    deltas = {
        str(seed): summary["cue"]["seed_macro_domain_accuracies"][str(seed)]
        - summary["rsc"]["seed_macro_domain_accuracies"][str(seed)]
        for seed in seeds
    }
    result = {
        "schema": 1,
        "runs": len(records),
        "seeds": seeds,
        "methods": summary,
        "cue_minus_rsc_by_seed": deltas,
        "cue_minus_rsc_mean": statistics.mean(deltas.values()),
        "ranking": sorted(
            METHODS, key=lambda method: summary[method]["mean_accuracy"], reverse=True
        ),
        "uncertainty": "SD across seed-level averages of the four target domains; three seeds do not establish broad superiority",
    }
    atomic_json(directory / "comparison.json", result)
    rows = [
        "| Method | Art | Cartoon | Photo | Sketch | Mean | Seed SD | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in result["ranking"]:
        value = summary[method]
        numbers = [f"{100 * value['domain_mean_accuracies'][domain]:.2f}" for domain in DOMAINS]
        sd = f"{100 * value['seed_sd']:.2f}" if value["seed_sd"] is not None else "n/a"
        rows.append(
            "| "
            + " | ".join(
                [method.upper()]
                + numbers
                + [
                    f"{100 * value['mean_accuracy']:.2f}",
                    sd,
                    f"{value['mean_nll']:.4f}",
                    f"{value['mean_ece']:.4f}",
                ]
            )
            + " |"
        )
    (directory / "comparison.md").write_text(
        "Accuracy and SD in percentage points; higher accuracy is better, lower NLL/ECE is better.\n\n"
        + "\n".join(rows)
        + "\n\n"
        + result["uncertainty"]
        + "\n"
    )
    return result


def main(project_root):
    """Dispatch explicit dataset preparation, verification, training, and testing commands."""
    root = Path(project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads; fixed across arms")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "smoke", "pretrained-smoke"):
        commands.add_parser(command)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=root / "data" / "PACS")
    prepare.add_argument("--manifest", type=Path, default=root / "data" / "pacs_manifest.json")
    prepare.add_argument("--archive", type=Path)
    prepare.add_argument("--download", action="store_true")
    data_smoke = commands.add_parser("data-smoke")
    data_smoke.add_argument("--manifest", type=Path, default=root / "data" / "pacs_manifest.json")
    data_smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    data_smoke.add_argument("--target", choices=DOMAINS)
    data_smoke.add_argument("--seed", type=int)
    data_smoke.add_argument("--steps", type=int)
    data_smoke.add_argument("--config", type=Path, default=root / "configs" / "pacs_resnet18.json")
    for command in ("train", "matrix", "profile"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, default=root / "configs" / "pacs_resnet18.json")
        if command != "profile":
            sub.add_argument("--manifest", type=Path, default=root / "data" / "pacs_manifest.json")
        if command == "train":
            sub.add_argument("--run-dir", type=Path, required=True)
            sub.add_argument("--method", choices=METHODS)
            sub.add_argument("--target", choices=DOMAINS)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after", type=int)
        elif command == "matrix":
            sub.add_argument("--runs-dir", type=Path, default=root / "runs" / "pacs_resnet18")
            sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            sub.add_argument("--evaluate", action="store_true")
        else:
            sub.add_argument(
                "--output", type=Path, default=root / "verification" / "cuda_memory.json"
            )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, default=root / "data" / "pacs_manifest.json")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "check":
        result = check(root)
    elif args.command == "smoke":
        result = smoke(root)
    elif args.command == "pretrained-smoke":
        result = pretrained_smoke(root)
    elif args.command == "data-smoke":
        config = load_config(args)
        if args.device == "cpu":
            config = replace(
                config,
                device="cpu",
                precision="fp32",
                batch_per_domain=1,
                eval_batch_size=3,
                steps=1,
                eval_every=1,
                workers=0,
                debug=True,
            ).validate()
        elif config.device != "cuda":
            parser.error("data-smoke --device cuda requires device=cuda in its config")
        steps = args.steps if args.steps is not None else (10 if args.device == "cuda" else 1)
        result = real_data_smoke(args.manifest, root, config, steps)
    elif args.command == "prepare":
        if args.download and args.archive:
            parser.error("Choose --download or --archive")
        archive = args.archive
        download_provenance = {}
        if args.download:
            archive = root / "data" / "PACS.zip"
            download_provenance = download_pacs(archive)
        if archive:
            provenance = extract_archive(archive, args.data_root)
            atomic_json(
                args.manifest.with_suffix(".archive.json"), {**provenance, **download_provenance}
            )
        manifest = prepare_manifest(
            args.data_root, args.manifest, root / "references" / "pacs_splits"
        )
        result = {
            "dataset_id": manifest["dataset_id"],
            "images": len(manifest["images"]),
            "manifest": str(args.manifest),
        }
    elif args.command == "train":
        result = train(
            load_config(args), args.manifest, args.run_dir, root, args.resume, args.stop_after
        )
    elif args.command == "evaluate":
        result = evaluate_target(args.manifest, args.run_dir, root)
    elif args.command == "profile":
        result = profile_memory(load_config(args), root)
        atomic_json(args.output, result)
    elif args.command == "matrix":
        result = matrix(
            load_config(args), args.manifest, args.runs_dir, root, args.seeds, args.evaluate
        )
    else:
        result = report_matrix(args.runs_dir)
    import json

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "profile" and not result["measured"]:
        raise SystemExit(2)
    if args.command == "profile" and not result["passes_budget"]:
        raise SystemExit(1)
