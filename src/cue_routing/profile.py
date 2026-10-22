"""Actual CUDA memory profiling at the unchanged benchmark batch and image size."""

import gc
from dataclasses import replace

import torch

from .config import METHODS
from .engine import autocast_context, make_model, optimize_batch, validate_device, vram_stats
from .io import environment, seed_everything, source_hash


def measure_method(current, project_root, measured_steps):
    """Keep each arm's allocations local so failure cleanup releases its entire graph."""
    seed_everything(current.seed, current.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make_model(current, project_root, initialize=False).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=current.learning_rate, weight_decay=current.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=current.precision == "fp16")
    images = torch.randn(
        3 * current.batch_per_domain, 3, current.image_size, current.image_size, device="cuda"
    )
    targets = torch.arange(len(images), device="cuda") % 7
    for _ in range(3):
        optimize_batch(model, optimizer, scaler, images, targets, current)
    torch.cuda.synchronize()
    for _ in range(measured_steps):
        optimize_batch(model, optimizer, scaler, images, targets, current)
    training = vram_stats(current)
    optimizer.zero_grad(set_to_none=True)
    del images, targets
    model.eval()
    with torch.no_grad(), autocast_context(current):
        validation_images = torch.randn(
            current.eval_batch_size, 3, current.image_size, current.image_size, device="cuda"
        )
        predictions = model(validation_images)
        if not torch.isfinite(predictions).all():
            raise FloatingPointError("Nonfinite validation predictions during memory profiling")
    combined = vram_stats(current)
    return {
        "status": "passed",
        "training": training,
        "training_and_validation": combined,
        "training_batch": 3 * current.batch_per_domain,
        "validation_batch": current.eval_batch_size,
        "warmup_steps": 3,
        "measured_steps": measured_steps,
        "random_weights": True,
        "input_kind": "synthetic_shape_matched_memory_probe",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def profile_memory(config, project_root, measured_steps=3):
    """Measure all arms after Adam allocation; never infer CUDA fit from a CPU smoke."""
    if measured_steps < 3:
        raise ValueError("At least three measured optimizer steps are required")
    config.validate()
    seed_everything(config.seed, config.device if torch.cuda.is_available() else "cpu")
    result = {
        "environment": environment(),
        "config": config.to_dict(),
        "source_sha256": source_hash(project_root),
        "measured": False,
        "methods": {},
        "budget_gib": config.max_vram_gib,
    }
    if not torch.cuda.is_available():
        result["reason"] = "CUDA unavailable on this host; 24 GB fit remains unmeasured"
        return result
    validate_device(config)
    if config.device != "cuda":
        raise ValueError("Memory profiling requires device=cuda")
    result["measured"] = True
    result["free_before_gib"] = torch.cuda.mem_get_info()[0] / 2**30
    for method in METHODS:
        current = replace(config, method=method)
        try:
            result["methods"][method] = measure_method(current, project_root, measured_steps)
        except (RuntimeError, FloatingPointError) as error:
            result["methods"][method] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        gc.collect()
        torch.cuda.empty_cache()
    result["passes_budget"] = all(
        value["status"] == "passed"
        and value["training_and_validation"]["peak_reserved_gib"] <= config.max_vram_gib
        for value in result["methods"].values()
    )
    return result
