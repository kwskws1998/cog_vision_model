"""Office-Home training and sealed evaluation using the unchanged shared AMP operators."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from cue_routing.engine import evaluate_loader, optimize_batch, validate_device, vram_stats
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
from cue_routing.metrics import aggregate_domains
from cue_routing.models import ResNet18Classifier
from cue_routing.protocol import is_better, record_target_access, seal_selection, verify_selection

from .config import CLASSES, DOMAINS, ExperimentConfig
from .data import evaluation_loader, load_manifest, training_loader, validate_manifest_config
from .fingerprint import source_hash


def make_model(config, project_root, initialize=True):
    """Initialize a common architecture, with optional downloaded or local pretraining."""
    return ResNet18Classifier(
        num_classes=len(CLASSES),
        grid_size=config.grid_size,
        pretrained=config.pretrained if initialize else False,
        pretrained_path=config.pretrained_path if initialize else "",
        cache_dir=Path(project_root) / ".cache" / "torch",
        freeze_bn=config.freeze_bn,
    ).to(config.device)


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
        "class_order": list(CLASSES),
        "domains": list(DOMAINS),
        "image_variant": manifest["variant"],
        "split_policy": manifest["split_policy"],
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
    manifest = load_manifest(manifest_path)
    validate_manifest_config(manifest, config)
    if manifest["kind"] != "officehome" and not config.debug:
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
                            "benchmark_evidence": manifest["kind"] == "officehome"
                            and not config.debug,
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
                    "session_seconds": time.monotonic() - began,
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
                eta_seconds = record["session_seconds"] * (end - step) / (step - start)
                print(
                    f"{config.method} target={config.target} seed={config.seed} step={step}/{config.steps} source_accuracy={validation['macro_accuracy']:.4f} loss={record['training']['loss']:.4f} elapsed_s={record['session_seconds']:.1f} eta_s={eta_seconds:.1f}",
                    flush=True,
                )
                interval = []
        summary = {
            "status": "complete" if end == config.steps else "interrupted_at_requested_boundary",
            "benchmark_evidence": manifest["kind"] == "officehome" and not config.debug,
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
        manifest = load_manifest(manifest_path)
        validate_manifest_config(manifest, config)
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
            "benchmark_evidence": manifest["kind"] == "officehome" and not config.debug,
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
