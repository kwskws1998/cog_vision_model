"""Matched training, exact checkpoint resume, and source-sealed final evaluation."""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import DOMAINS, ExperimentConfig
from .data import evaluation_loader, load_manifest, training_loader
from .io import (
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
    source_hash,
    state_hash,
)
from .metrics import aggregate_domains, classification_metrics
from .models import ResNet18Classifier
from .protocol import is_better, record_target_access, seal_selection, verify_selection
from .routing import routed_loss
from .rsc import rsc_logits


def validate_device(config):
    """Fail explicitly rather than silently changing device, precision, or batch size."""
    config.validate()
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use an explicit debug CPU configuration for verification"
            )
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support BF16")


def autocast_context(config):
    """Apply AMP only to the encoder; readouts disable it explicitly."""
    if config.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(
        "cuda", dtype=torch.float16 if config.precision == "fp16" else torch.bfloat16
    )


def make_model(config, project_root, initialize=True):
    """Initialize a common architecture, with optional downloaded or local pretraining."""
    return ResNet18Classifier(
        grid_size=config.grid_size,
        pretrained=config.pretrained if initialize else False,
        pretrained_path=config.pretrained_path if initialize else "",
        cache_dir=Path(project_root) / ".cache" / "torch",
        freeze_bn=config.freeze_bn,
    ).to(config.device)


def training_objective(model, images, targets, config):
    """Expose precisely one method-specific training operator on a shared backbone."""
    with autocast_context(config):
        maps = model.forward_features(images)
    if config.method == "cue":
        logits, weights = model.readout.cue_logits(maps)
        loss, predictions, diagnostics = routed_loss(
            logits,
            targets,
            weights,
            config.gate_tau,
            config.gate_temperature,
            config.joint_floor,
            config.routing,
            config.fixed_joint,
        )
    elif config.method == "rsc":
        features = maps.float().mean(dim=(2, 3))
        challenged, predictions, _, diagnostics = rsc_logits(
            features, targets, model.classifier, config.rsc_feature_drop, config.rsc_batch_drop
        )
        loss = F.cross_entropy(challenged, targets)
    else:
        predictions = model.readout(maps)
        loss, diagnostics = F.cross_entropy(predictions, targets), {}
    diagnostics["unmasked_train_accuracy"] = float(
        predictions.detach().argmax(dim=1).eq(targets).float().mean()
    )
    return loss, diagnostics


def optimize_batch(model, optimizer, scaler, images, targets, config):
    """Complete one update, retrying FP16 overflow on identical inputs and RNG state.

    GradScaler skips each overflowing attempt and reduces its scale. Retrying the
    same batch preserves the scheduled number of optimizer updates. Model buffers
    and RNGs are restored before replay, so BatchNorm and stochastic operators see
    the same state as a single attempt. Sixteen backoffs bound recovery; nonfinite
    losses and gradients without an enabled scaler remain fatal.
    """
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
            raise FloatingPointError(
                "Nonfinite training loss; AMP scale cannot repair the forward pass"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite = any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if nonfinite:
            if not enabled:
                raise FloatingPointError("Nonfinite gradients without enabled FP16 loss scaling")
            previous_scale = scaler.get_scale()
            if retries < 16:
                scaler.step(optimizer)
                scaler.update()
            current_scale = scaler.get_scale()
            if (
                retries == 16
                or not math.isfinite(current_scale)
                or not 0 < current_scale < previous_scale
            ):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Nonfinite gradients after {retries + 1} AMP attempts "
                    f"(method={config.method}, scale={current_scale}); no batch update completed"
                )
            del loss
            continue
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
        scaler.step(optimizer)
        scaler.update()
        return {
            "loss": float(loss.detach()),
            **diagnostics,
            "amp_overflow_retries": retries,
            "amp_scale": scaler.get_scale(),
        }
    raise AssertionError("AMP retry loop did not return or fail")


def vram_stats(config):
    """Enforce measured process reservation with headroom inside a 24 GB-class GPU."""
    if config.device != "cuda":
        return {"measured": False, "reason": "cpu_execution"}
    torch.cuda.synchronize()
    result = {
        "measured": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if result["peak_reserved_gib"] > config.max_vram_gib:
        raise RuntimeError(
            f"Measured peak reservation {result['peak_reserved_gib']:.3f} GiB exceeds the {config.max_vram_gib:.3f} GiB budget"
        )
    return result


@torch.no_grad()
def evaluate_loader(model, loader, config):
    """Always use the common unmasked inference rule for dataset-level metrics."""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        images = images.to(config.device, non_blocking=True)
        with autocast_context(config):
            logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
        vram_stats(config)
    if not all_logits:
        raise ValueError("Empty evaluation partition")
    logits, targets = torch.cat(all_logits), torch.cat(all_targets)
    return classification_metrics(logits, targets), logits, targets


def evaluate_sources(model, manifest, config):
    """Prevent the held-out target from entering checkpoint selection."""
    results = {}
    for domain in DOMAINS:
        if domain != config.target:
            loader = evaluation_loader(manifest, domain, "crossval", config)
            results[domain], _, _ = evaluate_loader(model, loader, config)
    return aggregate_domains(results)


def build_contract(config, manifest, project_root, model):
    """Record initial weights and all data, code, and configuration identities."""
    target_records = manifest["splits"][config.target]["test"]
    return {
        "schema": 2,
        "config": config.to_dict(),
        "dataset_id": manifest["dataset_id"],
        "dataset_kind": manifest["kind"],
        "target_partition_sha256": canonical_hash(target_records),
        "target_n": len(target_records),
        "source_sha256": source_hash(project_root),
        "initial_model_sha256": state_hash(model.state_dict()),
        "pretrained_file_sha256": file_hash(config.pretrained_path)
        if config.pretrained_path
        else None,
        "runtime": environment(),
        "selection_rule": "max_source_macro_accuracy_then_min_source_macro_nll_then_earliest_step",
    }


def finalize_run(run_dir, contract, best, best_state, summary):
    """Commit final weights and summary before exposing the selection seal to evaluation."""
    run_dir = Path(run_dir)
    if (run_dir / "selection.json").exists():
        seal = verify_selection(run_dir, contract)
        if seal["selection"] != best:
            raise ValueError("Completed checkpoint disagrees with the sealed selection")
    else:
        atomic_torch(
            run_dir / "best.pt",
            {
                "model": best_state,
                "step": best["step"],
                "selection": best,
                "contract_hash": canonical_hash(contract),
            },
        )
    atomic_json(run_dir / "summary.json", summary)
    if not (run_dir / "selection.json").exists():
        seal_selection(run_dir, contract, best, contract["config"]["steps"])


def train(config, manifest_path, run_dir, project_root, resume=False, stop_after=None):
    """Train without any target loader and seal selection only after all scheduled steps."""
    validate_device(config)
    project_root, run_dir = Path(project_root), Path(run_dir)
    manifest = load_manifest(manifest_path, project_root / "references" / "pacs_splits")
    if manifest["kind"] != "pacs" and not config.debug:
        raise ValueError("Artificial fixtures require debug=true")
    end = config.steps if stop_after is None else stop_after
    if not 0 < end <= config.steps or (end != config.steps and end % config.eval_every):
        raise ValueError("Intentional interruption must occur at a scheduled validation boundary")
    with run_lock(run_dir):
        if not resume and any(
            path.name not in (".writer.lock", ".running.json") for path in run_dir.iterdir()
        ):
            raise FileExistsError("Run directory is not empty; use --resume for an existing run")
        if resume and (run_dir / "target_access.json").exists():
            raise ValueError("Cannot continue training after target evaluation")
        seed_everything(config.seed, config.device)
        model = make_model(config, project_root)
        contract = build_contract(config, manifest, project_root, model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=config.device == "cuda" and config.precision == "fp16"
        )
        start, best, best_state, history = 0, None, None, []
        if resume:
            if read_json(run_dir / "contract.json") != contract:
                raise ValueError(
                    "Resume requires identical configuration, source, dataset, and initial weights"
                )
            if (run_dir / "last.pt").exists():
                checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
                if checkpoint["contract_hash"] != canonical_hash(contract):
                    raise ValueError("Checkpoint contract does not match this run")
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
                start, best, best_state, history = (
                    checkpoint["step"],
                    checkpoint["best"],
                    checkpoint["best_model"],
                    checkpoint["history"],
                )
                restore_rng(checkpoint["rng"])
                del checkpoint
                if (
                    not 0 < start <= config.steps
                    or not history
                    or history[-1]["step"] != start
                    or not 0 < best["step"] <= start
                ):
                    raise ValueError("Checkpoint step and selection history are inconsistent")
                atomic_json(run_dir / "history.json", history)
            elif any(
                (run_dir / name).exists()
                for name in ("history.json", "best.pt", "selection.json", "summary.json")
            ):
                raise ValueError("Saved artifacts exist but the resume checkpoint is missing")
            if start >= end:
                if start == config.steps:
                    existing = (
                        read_json(run_dir / "summary.json")
                        if (run_dir / "summary.json").exists()
                        else {}
                    )
                    summary = (
                        existing
                        if existing.get("status") == "complete"
                        and existing.get("completed_steps") == start
                        and existing.get("contract_hash") == canonical_hash(contract)
                        else {
                            "status": "complete",
                            "recovered_from_completed_checkpoint": True,
                            "completed_steps": start,
                            "best_source_selection": best,
                            "benchmark_evidence": manifest["kind"] == "pacs" and not config.debug,
                            "vram": history[-1]["vram"],
                            "contract_hash": canonical_hash(contract),
                        }
                    )
                    finalize_run(run_dir, contract, best, best_state, summary)
                    return summary
                raise ValueError("Resume endpoint must be after the saved step")
        else:
            atomic_json(run_dir / "contract.json", contract)
            atomic_json(run_dir / "environment.json", environment())
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loader = training_loader(manifest, config, start, end)
        interval, began = [], time.monotonic()
        for step, (images, targets) in enumerate(loader, start=start + 1):
            model.train()
            images, targets = (
                images.to(config.device, non_blocking=True),
                targets.to(config.device, non_blocking=True),
            )
            diagnostics = optimize_batch(model, optimizer, scaler, images, targets, config)
            interval.append(diagnostics)
            vram_stats(config)
            if step % config.eval_every == 0 or step == config.steps:
                validation = evaluate_sources(model, manifest, config)
                candidate = {
                    "step": step,
                    "macro_accuracy": validation["macro_accuracy"],
                    "macro_nll": validation["macro_nll"],
                }
                if is_better(candidate, best):
                    best = candidate
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                record = {
                    "step": step,
                    "amp_overflow_retries_total": sum(
                        item["amp_overflow_retries"] for item in interval
                    ),
                    "training": {
                        key: sum(item[key] for item in interval) / len(interval)
                        for key in diagnostics
                    },
                    "source_validation": validation,
                    "vram": vram_stats(config),
                }
                history.append(record)
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "best": best,
                    "best_model": best_state,
                    "history": history,
                    "rng": capture_rng(config.device),
                    "contract_hash": canonical_hash(contract),
                }
                atomic_torch(run_dir / "last.pt", checkpoint)
                atomic_json(run_dir / "history.json", history)
                print(
                    f"{config.method} target={config.target} seed={config.seed} step={step}/{config.steps} source_accuracy={validation['macro_accuracy']:.4f} loss={record['training']['loss']:.4f}",
                    flush=True,
                )
                interval = []
        summary = {
            "status": "complete" if end == config.steps else "interrupted_at_requested_boundary",
            "benchmark_evidence": manifest["kind"] == "pacs" and not config.debug,
            "completed_steps": end,
            "best_source_selection": best,
            "session_seconds": time.monotonic() - began,
            "vram": vram_stats(config),
            "contract_hash": canonical_hash(contract),
        }
        if end == config.steps:
            finalize_run(run_dir, contract, best, best_state, summary)
        else:
            atomic_json(run_dir / "summary.json", summary)
        return summary


def evaluate_target(manifest_path, run_dir, project_root):
    """Evaluate a sealed final checkpoint and save predictions for paired analysis."""
    run_dir, project_root = Path(run_dir), Path(project_root)
    with run_lock(run_dir):
        contract = read_json(run_dir / "contract.json")
        config = ExperimentConfig.from_dict(contract["config"])
        validate_device(config)
        seed_everything(config.seed, config.device)
        if environment() != contract["runtime"]:
            raise ValueError("Evaluation runtime differs from the training contract")
        if source_hash(project_root) != contract["source_sha256"]:
            raise ValueError("Executable sources changed since training")
        manifest = load_manifest(manifest_path, project_root / "references" / "pacs_splits")
        if manifest["dataset_id"] != contract["dataset_id"]:
            raise ValueError("Dataset differs from the training contract")
        seal = verify_selection(run_dir, contract)
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        if (
            checkpoint["contract_hash"] != canonical_hash(contract)
            or checkpoint["selection"] != seal["selection"]
        ):
            raise ValueError("Selected checkpoint metadata changed")
        model = make_model(config, project_root, initialize=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        record_target_access(run_dir, seal)
        loader = evaluation_loader(manifest, config.target, "test", config)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        metrics, logits, targets = evaluate_loader(model, loader, config)
        result = {
            "schema": 2,
            "target": config.target,
            "method": config.method,
            "seed": config.seed,
            "benchmark_evidence": manifest["kind"] == "pacs" and not config.debug,
            "metrics": metrics,
            "selection": seal,
            "vram": vram_stats(config),
            "contract_hash": canonical_hash(contract),
        }
        atomic_torch(
            run_dir / "target_predictions.pt",
            {
                "logits": logits,
                "targets": targets,
                "paths": [item[0] for item in loader.dataset.records],
                "contract_hash": canonical_hash(contract),
            },
        )
        result["prediction_sha256"] = file_hash(run_dir / "target_predictions.pt")
        atomic_json(run_dir / "target_metrics.json", result)
        return result
