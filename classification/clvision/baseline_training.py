"""Resumable FP32 reference CE training with explicit official-test provenance and measured ETA."""

import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

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
)

from .baseline_reference import ROOT, build_model, checked_reference, make_loaders
from .display import OPTIONS, BatchProgress, write
from .progress import duration


def initialize_rng(seed):
    """Add an explicit seed while using full FP32 and recording differences from historical defaults."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def synchronize(device):
    """Include completed GPU work in phase timings rather than only kernel launch overhead."""
    if device == "cuda":
        torch.cuda.synchronize()


def source_identity():
    """Bind resumes to only this runner, its support code, and the pinned author sources."""
    paths = [ROOT / "classification/reproduce_baseline.py", ROOT / "src/cue_routing/io.py"]
    paths += [
        Path(__file__).parent / name
        for name in ("baseline_reference.py", "baseline_training.py", "display.py", "progress.py")
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def contract_for(config, seed, data_identity):
    """Keep checkpoints from this full-training-set reference separate from CUE study checkpoints."""
    protocol = config.preview() | {"config": {**asdict(config), "seeds": [seed]}}
    if data_identity["kind"] != "official_cifar100":
        protocol.update(
            training_examples=data_identity["train_n"],
            evaluation_examples=data_identity["test_n"],
            evaluation_split="synthetic_test",
            synthetic_smoke=True,
            published_cifar100_reference_percent=None,
        )
    return {
        "schema": 1,
        "experiment": "eccv2022_author_cifar100_ce",
        "config": {**asdict(config), "seeds": [seed]},
        "seed": seed,
        "data": data_identity,
        "reference": checked_reference(),
        "source": source_identity(),
        "environment": environment(),
        "protocol": protocol,
    }


def phase(
    model,
    loader,
    device,
    optimizer=None,
    description="Test",
    save_predictions=False,
    split="official_test",
):
    """Train with final-head CE or evaluate all official test samples without changing model buffers."""
    model.train(optimizer is not None)
    synchronize(device)
    began = time.monotonic()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    sample_count, predictions, labels = 0, [], []
    with BatchProgress(description, len(loader), leave=False) as progress:
        with torch.set_grad_enabled(optimizer is not None):
            for step, (images, targets) in enumerate(loader, 1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)[0][0]
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite CE in {description}, step={step}")
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    ranking = logits.topk(5, dim=1).indices
                    matched = ranking.eq(targets[:, None])
                    totals[0] += loss.detach().double() * targets.numel()
                    totals[1] += matched[:, 0].sum()
                    totals[2] += matched.any(1).sum()
                    sample_count += targets.numel()
                    if save_predictions:
                        predictions.append(logits.detach().cpu())
                        labels.append(targets.cpu())
                progress.update(step)
                if optimizer is not None and step % OPTIONS.get().logging_steps == 0:
                    write(
                        f"[train] {description} batch={step}/{len(loader)} loss={loss.item():.4f}"
                    )
    synchronize(device)
    if sample_count != len(loader.dataset):
        raise ValueError("Evaluation/training did not visit the entire declared split")
    values = totals.cpu().tolist()
    seconds = time.monotonic() - began
    metrics = {
        "n": sample_count,
        "loss": values[0] / sample_count,
        "correct": int(values[1]),
        "accuracy": values[1] / sample_count,
        "top5_accuracy": values[2] / sample_count,
        "seconds": seconds,
        "images_per_second": sample_count / max(seconds, 1e-9),
    }
    outputs = (
        {
            "logits": torch.cat(predictions),
            "labels": torch.cat(labels),
            "sample_ids": torch.arange(sample_count),
            "split": split,
        }
        if save_predictions
        else None
    )
    return metrics, outputs


def train_seed(config, seed, train, test, identity, run_dir, resume=False, stop_after_epoch=None):
    """Checkpoint each completed epoch and resume with model, optimizer and RNG state together."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir):
        initialize_rng(seed)
        contract = contract_for(config, seed, identity)
        evaluation_split = contract["protocol"]["evaluation_split"]
        checkpoint_path = run_dir / "last.pt"
        exists = (run_dir / "contract.json").exists()
        if exists and not resume:
            raise FileExistsError(f"Existing run: {run_dir}; use --resume or a fresh --runs-dir")
        if exists and read_json(run_dir / "contract.json") != contract:
            raise ValueError("Resume contract differs: data, recipe, source or environment changed")
        if resume and not exists:
            raise FileNotFoundError(f"No resume contract in {run_dir}")
        model = build_model().to(config.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        train_loader, test_loader = make_loaders(train, test, config)
        history, elapsed_before, start, final_metrics = [], 0.0, 0, None
        if exists:
            if not checkpoint_path.exists():
                raise FileNotFoundError("No complete epoch checkpoint; choose a new run directory")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["contract_hash"] != canonical_hash(contract):
                raise ValueError("Checkpoint contract mismatch")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
            history, elapsed_before, start = saved["history"], saved["elapsed_s"], saved["epoch"]
        else:
            atomic_json(run_dir / "contract.json", contract)
        endpoint = (
            config.epochs if stop_after_epoch is None else min(config.epochs, stop_after_epoch)
        )
        write(
            f"[run] seed={seed} pid={os.getpid()} train_n={len(train)} test_n={len(test)} "
            f"train_steps={len(train_loader) * config.epochs} precision=fp32 "
            f"train_batch={config.batch_size} eval_batch={config.eval_batch_size} "
            f"schedule={config.schedule} elapsed={duration(elapsed_before)} remaining~=estimating"
        )
        began = time.monotonic()
        for epoch in range(start + 1, endpoint + 1):
            epoch_began = time.monotonic()
            lr = config.rate(epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_metrics, _ = phase(
                model,
                train_loader,
                config.device,
                optimizer,
                f"Train CIFAR100/ERM/s{seed} epoch={epoch}/{config.epochs}",
            )
            final_metrics, outputs = phase(
                model,
                test_loader,
                config.device,
                description=f"Official test CIFAR100/ERM/s{seed} epoch={epoch}/{config.epochs}",
                save_predictions=epoch == config.epochs,
                split=evaluation_split,
            )
            if outputs is not None:
                atomic_torch(run_dir / "test_predictions.pt", outputs)
            memory = (
                {
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
                if config.device == "cuda"
                else {}
            )
            row = {
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "official_test": final_metrics,
                **memory,
            }
            history.append(row)
            elapsed = elapsed_before + time.monotonic() - began
            atomic_torch(
                checkpoint_path,
                {
                    "contract_hash": canonical_hash(contract),
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(config.device),
                    "history": history,
                    "elapsed_s": elapsed,
                },
            )
            row_seconds = time.monotonic() - epoch_began
            remaining = row_seconds * (config.epochs - epoch)
            finish = datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(seconds=remaining)
            status = {
                "stage": "training",
                "epoch": epoch,
                "epochs": config.epochs,
                "elapsed_s": elapsed,
                "remaining_estimate_s": remaining,
                "finish_KST_estimate": finish.isoformat(),
                "latest": row,
            }
            atomic_json(run_dir / "history.json", history)
            atomic_json(run_dir / "status.json", status)
            write(
                f"[epoch] CIFAR100/ERM/s{seed} epoch={epoch}/{config.epochs} "
                f"lr={lr:g} train_loss={train_metrics['loss']:.4f} "
                f"train_accuracy={train_metrics['accuracy']:.4f} "
                f"test_loss={final_metrics['loss']:.4f} test_accuracy={final_metrics['accuracy']:.4f} "
                f"test_n={final_metrics['n']} images/s={train_metrics['images_per_second']:.1f} "
                f"peak_reserved={memory.get('peak_reserved_gib', 0):.2f} GiB "
                f"elapsed={duration(elapsed)} remaining~={duration(remaining)} "
                f"finish_KST~={finish.isoformat()}"
            )
        if len(history) != config.epochs:
            return {"status": "incomplete", "completed_epochs": len(history)}
        final_metrics, outputs = phase(
            model,
            test_loader,
            config.device,
            description=f"Final checkpoint test CIFAR100/ERM/s{seed}",
            save_predictions=True,
            split=evaluation_split,
        )
        expected = history[-1]["official_test"]
        if final_metrics["correct"] != expected["correct"] or not math.isclose(
            final_metrics["loss"], expected["loss"], rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("Final checkpoint evaluation differs from final-epoch test")
        atomic_torch(run_dir / "test_predictions.pt", outputs)
        atomic_torch(
            run_dir / "final.pt",
            {
                "model": model.state_dict(),
                "epoch": config.epochs,
                "contract_hash": canonical_hash(contract),
            },
        )
        best = max(history, key=lambda row: row["official_test"]["accuracy"])
        summary = {
            "status": "complete",
            "seed": seed,
            "epoch": config.epochs,
            "data_kind": identity["kind"],
            "evaluation_split": evaluation_split,
            "final_test": final_metrics,
            "best_observed_test": {
                **best["official_test"],
                "epoch": best["epoch"],
                "selection": "maximum over test evaluations; not a held-out selection",
            },
            "elapsed_s": elapsed_before + time.monotonic() - began,
            "contract_hash": canonical_hash(contract),
            "predictions_sha256": file_hash(run_dir / "test_predictions.pt"),
            "checkpoint_sha256": file_hash(run_dir / "final.pt"),
        }
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(
            run_dir / "status.json",
            {
                "stage": "complete",
                "epoch": config.epochs,
                "elapsed_s": summary["elapsed_s"],
                "remaining_estimate_s": 0,
            },
        )
        write(
            f"[complete] seed={seed} final_test={100 * final_metrics['accuracy']:.2f}% "
            f"best_observed_test={100 * best['official_test']['accuracy']:.2f}% "
            f"best_epoch={best['epoch']} test_n={final_metrics['n']} "
            f"elapsed={duration(summary['elapsed_s'])}"
        )
        return summary
