"""Identify the active GPU job and expose measured host waits without altering training inputs."""

import gc
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from cue_routing.io import atomic_json


def reset_run_memory(config):
    """Exclude inactive preflight/previous-run cache before measuring a new run's allocations."""
    if config.device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def diagnostic_scalars(loss, diagnostics):
    """Copy detached loss metrics to the host together instead of synchronizing once per scalar."""
    tensors = [loss.detach()]
    for value in diagnostics.values():
        tensors.append(
            value.detach() if isinstance(value, torch.Tensor) else loss.new_tensor(value)
        )
    values = torch.stack(tensors).detach().cpu().tolist()
    return dict(zip(("loss", *diagnostics), values))


def timed_batches(loader, timings):
    """Measure time blocked on loader creation, fetches, and teardown separately from batch work."""
    began = time.monotonic()
    iterator = iter(loader)
    timings["data_wait_seconds"] = timings.get("data_wait_seconds", 0) + time.monotonic() - began
    while True:
        began = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            timings["data_wait_seconds"] += time.monotonic() - began
            return
        timings["data_wait_seconds"] += time.monotonic() - began
        yield batch


def runtime_status(run_dir, config, stage, epoch=None, **details):
    """Persist the current PID, effective recipe, phase, and allocator counters for remote diagnosis."""
    value = {
        "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(Path(run_dir).resolve()),
        "stage": stage,
        "epoch": epoch,
        "dataset": config.dataset,
        "method": config.method,
        "activation": config.activation,
        "seed": config.seed,
        "train_batch_size": config.batch_size,
        "eval_batch_size": config.eval_batch_size,
        "precision": config.precision,
        "eval_precision": config.eval_precision,
        "workers": config.workers,
        "device": config.device,
        "vram_budget_gib": config.max_vram_gib,
        "vram": None,
        **details,
    }
    if config.device == "cuda":
        value["vram"] = {
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    atomic_json(Path(run_dir) / "runtime_status.json", value)
    return value
