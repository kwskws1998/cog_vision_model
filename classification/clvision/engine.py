"""Finite SGD updates, exact epoch resume, immutable final checkpoints, and sealed testing."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cue_routing.engine import autocast_context, validate_device, vram_stats
from cue_routing.io import (
    atomic_json,
    atomic_torch,
    canonical_hash,
    capture_rng,
    environment,
    file_hash,
    read_json,
    restore_rng,
    run_lock,
    seed_everything,
    state_hash,
)
from cue_routing.metrics import classification_metrics

from .config import CORRUPTIONS, Config
from .data import CorruptionDataset, load_manifest, make_loader
from .identity import source_hash
from .losses import training_objective
from .models import SmallResNet18


def make_model(config):
    """Construct the matched shared architecture and only the required Zipf auxiliary head."""
    return SmallResNet18(
        len_classes(config.dataset), auxiliary=config.method == "zipfls_dense2", seed=config.seed
    ).to(config.device)


def len_classes(dataset):
    """Resolve the fixed benchmark class count without inspecting any test labels."""
    return {"cifar100": 100, "tiny_imagenet": 200, "dominoes": 2}[dataset]


def optimizer_for(model, config):
    """Use the same SGD rule for all shared trainable tensors."""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def optimize_batch(model, optimizer, scaler, images, targets, config):
    """Replay AMP overflow with identical buffers and RNG, counting only accepted updates."""
    enabled = scaler.is_enabled()
    rng = capture_rng(config.device) if enabled else None
    buffers = (
        {name: value.detach().clone() for name, value in model.named_buffers()} if enabled else {}
    )
    for retries in range(17):
        if retries:
            restore_rng(rng)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = training_objective(model, images, targets, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite forward loss; no update was accepted")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        invalid = any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if invalid:
            if not enabled:
                raise FloatingPointError("Nonfinite gradients without FP16 scaling")
            previous = scaler.get_scale()
            if retries < 16:
                scaler.step(optimizer)
                scaler.update()
            current = scaler.get_scale()
            if retries == 16 or not math.isfinite(current) or not 0 < current < previous:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Persistent AMP overflow; no batch update completed")
            del loss
            continue
        if config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
        scaler.step(optimizer)
        scaler.update()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(
                "Nonfinite updated parameters; resume from the last epoch checkpoint"
            )
        if (
            config.device == "cuda"
            and torch.cuda.max_memory_reserved() > config.max_vram_gib * 2**30
        ):
            raise RuntimeError("Peak CUDA reservation exceeds the prespecified budget")
        return {
            "loss": float(loss.detach()),
            **diagnostics,
            "amp_overflow_retries": retries,
            "amp_scale": scaler.get_scale(),
        }
    raise AssertionError("AMP replay exhausted without a result")


@torch.no_grad()
def evaluate_loader(model, loader, config):
    """Keep BatchNorm frozen during evaluation and collect the same final-head predictions."""
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        with autocast_context(config):
            output = model(images.to(config.device, non_blocking=True))
        logits.append(output.cpu())
        labels.append(targets.long().cpu())
    if not logits:
        raise ValueError("Empty evaluation partition")
    logits, labels = torch.cat(logits), torch.cat(labels)
    return classification_metrics(logits, labels), logits, labels


def validate_manifest(config, manifest):
    """Forbid debug fixtures from producing production benchmark artifacts."""
    if manifest["dataset"] != config.dataset or len(manifest["classes"]) != len_classes(
        config.dataset
    ):
        raise ValueError("Configuration and dataset class mapping differ")
    if manifest["kind"] == "fixture" and not config.debug:
        raise ValueError("Synthetic datasets require debug=true")


def build_contract(config, manifest, model, root, phase, selection_id):
    """Bind initial shared weights, recipe, dataset, selection, sources, and execution environment."""
    return {
        "schema": 1,
        "config": config.to_dict(),
        "dataset_id": manifest["dataset_id"],
        "class_order": manifest["classes"],
        "source_sha256": source_hash(root),
        "initial_shared_model_sha256": model.shared_state_hash(),
        "initial_complete_model_sha256": state_hash(model.state_dict()),
        "phase": phase,
        "development_selection_sha256": selection_id,
        "selection_rule": "fixed_final_epoch",
        "runtime": environment(),
        "benchmark_evidence": manifest["kind"] != "fixture" and not config.debug,
        "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "inference_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("auxiliary.")
        ),
    }


def seal_run(run_dir, contract, model_state, history, accepted_steps, elapsed):
    """Expose completion only after final weights and the fixed-final validation score are durable."""
    run_dir = Path(run_dir)
    contract_id = canonical_hash(contract)
    atomic_torch(
        run_dir / "final.pt",
        {"model": model_state, "epoch": contract["config"]["epochs"], "contract_hash": contract_id},
    )
    atomic_json(run_dir / "history.json", history)
    summary = {
        "status": "complete",
        "completed_epochs": contract["config"]["epochs"],
        "accepted_optimizer_steps": accepted_steps,
        "final_validation": history[-1]["validation"],
        "elapsed_seconds": elapsed,
        "contract_hash": contract_id,
        "benchmark_evidence": contract["benchmark_evidence"],
        "vram": history[-1]["vram"],
    }
    atomic_json(run_dir / "summary.json", summary)
    completion = {
        "schema": 1,
        "contract_hash": contract_id,
        "final_checkpoint_sha256": file_hash(run_dir / "final.pt"),
        "history_sha256": file_hash(run_dir / "history.json"),
        "summary_sha256": file_hash(run_dir / "summary.json"),
        "completed_epochs": contract["config"]["epochs"],
        "selection_rule": "fixed_final_epoch",
    }
    atomic_json(run_dir / "completion.json", completion)
    return summary


def verify_run(run_dir, root, expected_config=None, dataset_id=None, selection_id=None):
    """Reject altered, incomplete, stale, or configuration-mismatched completed runs."""
    run_dir = Path(run_dir)
    contract = read_json(run_dir / "contract.json")
    config = Config.from_dict(contract["config"])
    completion = read_json(run_dir / "completion.json")
    if source_hash(root) != contract["source_sha256"]:
        raise ValueError("Executable sources changed since training")
    if expected_config is not None and config.to_dict() != expected_config.to_dict():
        raise ValueError("Completed run configuration mismatch")
    if dataset_id is not None and dataset_id != contract["dataset_id"]:
        raise ValueError("Completed run dataset mismatch")
    if selection_id is not None and selection_id != contract["development_selection_sha256"]:
        raise ValueError("Completed run development selection mismatch")
    checks = {
        "contract_hash": canonical_hash(contract),
        "completed_epochs": config.epochs,
        "final_checkpoint_sha256": file_hash(run_dir / "final.pt"),
        "history_sha256": file_hash(run_dir / "history.json"),
        "summary_sha256": file_hash(run_dir / "summary.json"),
        "selection_rule": "fixed_final_epoch",
    }
    if any(completion.get(key) != value for key, value in checks.items()):
        raise ValueError("Completed training artifacts changed")
    summary = read_json(run_dir / "summary.json")
    history = read_json(run_dir / "history.json")
    if summary.get("status") != "complete" or summary.get("completed_epochs") != config.epochs:
        raise ValueError("Incomplete training summary")
    if (
        not history
        or history[-1]["epoch"] != config.epochs
        or summary["final_validation"] != history[-1]["validation"]
    ):
        raise ValueError("Final validation metadata is inconsistent")
    return contract, completion, summary


def train(
    config, manifest, run_dir, root, phase="main", selection_id="", resume=False, stop_after=None
):
    """Train whole matched epochs using validation only, with exactly resumable SGD state."""
    validate_device(config)
    if not isinstance(manifest, dict):
        manifest = load_manifest(manifest)
    validate_manifest(config, manifest)
    if phase not in ("development", "main", "mechanism", "debug"):
        raise ValueError("Unknown experiment phase")
    endpoint = config.epochs if stop_after is None else stop_after
    if type(endpoint) is not int or not 0 < endpoint <= config.epochs:
        raise ValueError("Interruption must be a complete declared epoch")
    run_dir, root = Path(run_dir), Path(root)
    with run_lock(run_dir):
        content = [
            path for path in run_dir.iterdir() if path.name not in (".writer.lock", ".running.json")
        ]
        if content and not resume:
            raise FileExistsError("Run exists; explicit resume/reuse is required")
        if (run_dir / "test_access.json").exists():
            if not (run_dir / "completion.json").exists() or not resume:
                raise ValueError("Training after final-test access is forbidden")
        seed_everything(config.seed, config.device)
        model = make_model(config)
        contract = build_contract(config, manifest, model, root, phase, selection_id)
        if (run_dir / "completion.json").exists() and resume:
            old = read_json(run_dir / "contract.json")
            if old != contract:
                raise ValueError("Completed run identity differs from this requested run")
            return verify_run(run_dir, root, config, manifest["dataset_id"], selection_id)[2]
        optimizer = optimizer_for(model, config)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=config.device == "cuda" and config.precision == "fp16"
        )
        start, history, steps, previous_seconds = 0, [], 0, 0.0
        if resume and (run_dir / "contract.json").exists():
            if read_json(run_dir / "contract.json") != contract:
                raise ValueError(
                    "Resume requires identical data, source, runtime, configuration, and initial weights"
                )
            if (run_dir / "last.pt").exists():
                checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
                if checkpoint["contract_hash"] != canonical_hash(contract):
                    raise ValueError("Resume checkpoint identity mismatch")
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
                start, history, steps, previous_seconds = (
                    checkpoint["epoch"],
                    checkpoint["history"],
                    checkpoint["steps"],
                    checkpoint["elapsed_seconds"],
                )
                restore_rng(checkpoint["rng"])
                del checkpoint
                if (
                    not history
                    or history[-1]["epoch"] != start
                    or len(history) != start
                    or not 0 < start <= config.epochs
                ):
                    raise ValueError("Invalid resume epoch or validation history")
                atomic_json(run_dir / "history.json", history)
            elif any(
                (run_dir / name).exists() for name in ("history.json", "final.pt", "summary.json")
            ):
                raise ValueError("Checkpoint missing from a partially saved run")
        else:
            atomic_json(run_dir / "contract.json", contract)
        if start == config.epochs:
            return seal_run(run_dir, contract, model.state_dict(), history, steps, previous_seconds)
        if endpoint <= start:
            raise ValueError("Resume endpoint must follow the saved epoch")
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        for epoch in range(start + 1, endpoint + 1):
            rate = config.learning_rate * config.lr_decay ** sum(
                epoch > milestone for milestone in config.milestones
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            totals, count, images_seen, retries = {}, 0, 0, 0
            epoch_began = time.monotonic()
            loader = make_loader(manifest, config, "train", epoch)
            for images, targets in loader:
                result = optimize_batch(
                    model,
                    optimizer,
                    scaler,
                    images.to(config.device),
                    targets.to(config.device),
                    config,
                )
                for key, value in result.items():
                    totals[key] = totals.get(key, 0.0) + value * len(targets)
                count += 1
                steps += 1
                images_seen += len(targets)
                retries += result["amp_overflow_retries"]
            if count == 0:
                raise ValueError("Empty training epoch")
            if count != len(loader) or (
                not config.debug_max_batches and images_seen != len(loader.dataset)
            ):
                raise ValueError("Training epoch omitted or duplicated scheduled batches")
            train_seconds = time.monotonic() - epoch_began
            validation = evaluate_loader(
                model, make_loader(manifest, config, "validation"), config
            )[0]
            validation = {key: value for key, value in validation.items() if key != "confusion"}
            elapsed = previous_seconds + time.monotonic() - began
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": rate,
                    "accepted_optimizer_steps": steps,
                    "training": {key: value / images_seen for key, value in totals.items()},
                    "epoch_optimizer_steps": count,
                    "epoch_training_images": images_seen,
                    "amp_overflow_retries_total": retries,
                    "validation": validation,
                    "training_images_per_second": images_seen / max(train_seconds, 1e-9),
                    "elapsed_seconds": elapsed,
                    "vram": vram_stats(config),
                }
            )
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "steps": steps,
                "elapsed_seconds": elapsed,
                "rng": capture_rng(config.device),
                "contract_hash": canonical_hash(contract),
            }
            atomic_torch(run_dir / "last.pt", checkpoint)
            atomic_json(run_dir / "history.json", history)
            eta = (time.monotonic() - began) * (config.epochs - epoch) / (epoch - start)
            print(
                f"{config.dataset} {config.method} seed={config.seed} epoch={epoch}/{config.epochs} val_accuracy={validation['accuracy']:.4f} loss={history[-1]['training']['loss']:.4f} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                flush=True,
            )
        if endpoint == config.epochs:
            return seal_run(
                run_dir,
                contract,
                model.state_dict(),
                history,
                steps,
                previous_seconds + time.monotonic() - began,
            )
        return {
            "status": "interrupted_at_epoch_boundary",
            "completed_epochs": endpoint,
            "accepted_optimizer_steps": steps,
        }


def load_final(run_dir, root, manifest):
    """Load only a verified completed checkpoint in its original execution environment."""
    contract, completion, _ = verify_run(run_dir, root, dataset_id=manifest["dataset_id"])
    config = Config.from_dict(contract["config"])
    validate_device(config)
    seed_everything(config.seed, config.device)
    if environment() != contract["runtime"]:
        raise ValueError("Evaluation runtime differs from training")
    checkpoint = torch.load(Path(run_dir) / "final.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["contract_hash"] != canonical_hash(contract)
        or checkpoint["epoch"] != config.epochs
    ):
        raise ValueError("Final checkpoint metadata differs from the completion seal")
    model = make_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, contract, completion


def authorize_run(run_dir, study_seal):
    """Require the immutable whole-study authorization before constructing a test loader."""
    run_dir = Path(run_dir).resolve()
    if study_seal.get("purpose") != "final_evaluation_after_all_prespecified_training":
        raise ValueError("Final testing requires a sealed study")
    row = study_seal["runs"].get(str(run_dir))
    if row is None or file_hash(run_dir / "completion.json") != row:
        raise ValueError("Run is missing or changed in the study seal")
    contract = read_json(run_dir / "contract.json")
    if (
        contract["source_sha256"] != study_seal["source_sha256"]
        or contract["phase"] == "development"
    ):
        raise ValueError("Development runs cannot access final tests")


def record_access(run_dir, seal, kind):
    """Journal explicit final-test access separately from all validation-only artifacts."""
    path = Path(run_dir) / "test_access.json"
    events = read_json(path) if path.exists() else []
    events.append(
        {"event": len(events) + 1, "kind": kind, "study_seal_sha256": canonical_hash(seal)}
    )
    atomic_json(path, events)


def dominoes_metrics(logits, labels):
    """Weight the same counterfactual vehicle pairs into all three prespecified regimes."""
    if len(labels) % 2 or not torch.equal(labels[::2], labels[1::2]):
        raise ValueError("Counterfactual vehicle pairs are not aligned")
    prediction = logits.argmax(1).reshape(-1, 2)
    truth = labels[::2]
    correct = prediction.eq(truth[:, None]).double()
    rows = torch.arange(len(truth))
    aligned, reversed_ = correct[rows, truth], correct[rows, 1 - truth]
    groups = {
        f"vehicle{vehicle}_digit{digit}": float(correct[truth == vehicle, digit].mean())
        for vehicle in (0, 1)
        for digit in (0, 1)
    }
    return {
        "accuracy_by_digit_agreement": {
            str(value): float((value * aligned + (1 - value) * reversed_).mean())
            for value in (0.95, 0.5, 0.05)
        },
        "group_accuracies": groups,
        "worst_group_accuracy": min(groups.values()),
        "digit_counterfactual_prediction_flip_rate": float(
            prediction[:, 0].ne(prediction[:, 1]).double().mean()
        ),
        "vehicle_images": len(truth),
        "counterfactual_images": len(labels),
    }


def evaluate_run(manifest, run_dir, root, study_seal):
    """Open the final partition only after whole-study completion, saving aligned predictions."""
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        authorize_run(run_dir, study_seal)
        contract, _, _ = verify_run(run_dir, root, dataset_id=manifest["dataset_id"])
        artifact = run_dir / "test_metrics.json"
        if artifact.exists():
            existing = read_json(artifact)
            if (
                existing["contract_hash"] != canonical_hash(contract)
                or existing["study_seal_sha256"] != canonical_hash(study_seal)
                or existing["prediction_sha256"] != file_hash(run_dir / "test_predictions.pt")
            ):
                raise ValueError("Previously evaluated artifacts changed")
            return existing
        model, config, _, _ = load_final(run_dir, root, manifest)
        record_access(run_dir, study_seal, "clean_test")
        loader = make_loader(manifest, config, "test")
        began = time.monotonic()
        metrics, logits, labels = evaluate_loader(model, loader, config)
        duration = time.monotonic() - began
        result = {
            "dataset": config.dataset,
            "method": config.method,
            "seed": config.seed,
            "metrics": metrics,
            "contract_hash": canonical_hash(contract),
            "study_seal_sha256": canonical_hash(study_seal),
            "benchmark_evidence": contract["benchmark_evidence"],
            "inference_images_per_second": len(labels) / max(duration, 1e-9),
        }
        if config.dataset == "dominoes":
            result["cue_conflict"] = dominoes_metrics(logits, labels)
            result["panel_diagnostics"] = {
                name: evaluate_loader(
                    model, make_loader(manifest, config, "test", diagnostic=name), config
                )[0]
                for name in ("vehicle_only", "digit_only")
            }
        atomic_torch(
            run_dir / "test_predictions.pt",
            {
                "logits": logits,
                "targets": labels,
                "sample_ids": [
                    loader.dataset.sample_id(index) for index in range(len(loader.dataset))
                ],
                "contract_hash": canonical_hash(contract),
            },
        )
        result["prediction_sha256"] = file_hash(run_dir / "test_predictions.pt")
        atomic_json(artifact, result)
        return result


def evaluate_corruptions(corruptions, clean, run_dir, root, study_seal):
    """Evaluate all75 fixed test conditions with immutable checkpoints and resumable cells."""
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        authorize_run(run_dir, study_seal)
        model, config, contract, _ = load_final(run_dir, root, clean)
        if config.dataset != "cifar100":
            raise ValueError("CIFAR corruptions require the matching CIFAR100 model")
        identity = {
            "contract_hash": canonical_hash(contract),
            "study_seal_sha256": canonical_hash(study_seal),
            "dataset_id": corruptions["dataset_id"],
        }
        path = run_dir / "corruption_cells.json"
        existing = read_json(path) if path.exists() else {**identity, "cells": {}}
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Corruption evaluation identity changed")
        record_access(run_dir, study_seal, "cifar100_c")
        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                key = f"{corruption}/{severity}"
                if key in existing["cells"]:
                    continue
                dataset = CorruptionDataset(corruptions, clean, corruption, severity)
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size,
                    num_workers=config.workers,
                    shuffle=False,
                    generator=torch.Generator().manual_seed(0),
                )
                metrics = evaluate_loader(model, loader, config)[0]
                existing["cells"][key] = {
                    name: value for name, value in metrics.items() if name != "confusion"
                }
                atomic_json(path, existing)
                print(
                    f"CIFAR100-C {config.method} seed={config.seed} {key} accuracy={metrics['accuracy']:.4f}",
                    flush=True,
                )
        if set(existing["cells"]) != {
            f"{name}/{level}" for name in CORRUPTIONS for level in range(1, 6)
        }:
            raise ValueError("Corruption grid is incomplete or includes undeclared cells")
        result = {
            **identity,
            "cells": existing["cells"],
            "mean_corruption_accuracy": sum(
                value["accuracy"] for value in existing["cells"].values()
            )
            / 75,
            "metric_definition": "Unnormalized mean accuracy over15 corruptions x5 severities, not normalized mCE",
        }
        atomic_json(run_dir / "corruption_metrics.json", result)
        return result
