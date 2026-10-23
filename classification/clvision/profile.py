"""Measure real full-batch training and evaluation before permitting the main study."""

import gc
import hashlib
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from itertools import chain, islice

import torch

from cue_routing.engine import validate_device, vram_stats
from cue_routing.io import atomic_json, canonical_hash, environment, seed_everything

from .config import MAIN_METHODS, MECHANISM_METHODS
from .data import make_loader
from .display import BatchProgress
from .engine import evaluate_loader, make_model, optimize_batch, optimizer_for
from .identity import source_hash


def memory_signature(config):
    """Ignore scalar loss coefficients while binding all memory-relevant recipe fields."""
    return {
        key: getattr(config, key)
        for key in (
            "dataset",
            "tiny_imagenet_size",
            "activation",
            "initialization",
            "normalization",
            "method",
            "batch_size",
            "eval_batch_size",
            "device",
            "precision",
            "eval_precision",
            "max_vram_gib",
            "debug",
        )
    }


@contextmanager
def profile_iterator(loader):
    """Close partial PyTorch 2.8 loaders on success or failure, including their pin-memory thread."""
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def profile_method(manifest, config, steps, activity):
    """Separate loader startup and warmup from steady throughput while retaining all safety checks."""
    if steps < 10:
        raise ValueError("At least 10 accepted real-data updates are required for timing")
    validate_device(config)
    validate_device(replace(config, precision=config.eval_precision))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seed_everything(config.seed, config.device)
    model = make_model(config)
    initial = model.shared_state_hash()
    before = next(model.parameters()).detach().clone()
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    first_batch = None
    model.train()
    activity["stage"] = "training"
    warmup_steps = 5
    measured_images = 0
    loader_began = time.monotonic()
    loader = make_loader(manifest, config, "train", epoch=1)
    with ExitStack() as stack:
        progress = stack.enter_context(
            BatchProgress(f"Preflight {config.dataset}/{config.method}", steps)
        )
        iterator = stack.enter_context(profile_iterator(loader))
        for index in range(steps):
            if index == warmup_steps:
                torch.cuda.synchronize()
                began = time.monotonic()
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = stack.enter_context(profile_iterator(loader))
                images, labels = next(iterator)
            if len(images) != config.batch_size:
                raise ValueError("Preflight requires the full declared training batch")
            if index == 0:
                training_startup = time.monotonic() - loader_began
                first_batch = canonical_hash(
                    {
                        "images": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                        "labels": labels.tolist(),
                    }
                )
            diagnostics = optimize_batch(
                model,
                optimizer,
                scaler,
                images.to(config.device, non_blocking=True),
                labels.to(config.device, non_blocking=True),
                config,
            )
            activity["accepted_optimizer_steps"] = index + 1
            progress.update(index + 1, stage="warmup" if index < warmup_steps else "measured")
            if index >= warmup_steps:
                measured_images += len(images)
        torch.cuda.synchronize()
        seconds = time.monotonic() - began
    activity["stage"] = "validation"
    loader_began = time.monotonic()
    with profile_iterator(make_loader(manifest, config, "validation")) as iterator:
        validation_batch = next(iterator)
        evaluation_startup = time.monotonic() - loader_began
        if len(validation_batch[0]) != config.eval_batch_size:
            raise ValueError("Preflight requires the full declared evaluation batch")
        evaluate_loader(model, [validation_batch], config)
        torch.cuda.synchronize()
        evaluation_began = time.monotonic()
        metrics, _, _ = evaluate_loader(
            model, chain((validation_batch,), islice(iterator, 9)), config
        )
        torch.cuda.synchronize()
        evaluation_seconds = time.monotonic() - evaluation_began
    return {
        "signature": memory_signature(config),
        "initial_shared_model_sha256": initial,
        "first_augmented_batch_sha256": first_batch,
        "accepted_optimizer_steps": steps,
        "backbone_updated": not torch.equal(before, next(model.parameters()).detach()),
        "training_images_per_second": measured_images / max(seconds, 1e-9),
        "training_loader_startup_seconds": training_startup,
        "training_warmup_steps": warmup_steps,
        "training_measured_steps": steps - warmup_steps,
        "training_measured_images": measured_images,
        "evaluation_images_per_second": metrics["n"] / max(evaluation_seconds, 1e-9),
        "evaluation_loader_startup_seconds": evaluation_startup,
        "evaluation_warmup_batches": 1,
        "evaluation_measured_images": metrics["n"],
        "evaluation_reuses_first_batch_after_warmup": True,
        "vram": vram_stats(config),
        "last_diagnostics": diagnostics,
        "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def selected_methods(config, methods):
    """Allow an explicit subset without silently profiling unrelated training interventions."""
    available = MECHANISM_METHODS if config.dataset == "dominoes" else MAIN_METHODS
    if methods is None:
        return available
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(available):
        raise ValueError("Preflight methods must be a nonempty unique supported subset")
    return tuple(methods)


def profile_dataset(root, manifest, base_config, destination, steps=10, methods=None):
    """Use actual train/validation images, SGD momentum, AMP, and every selected intervention."""
    methods = selected_methods(base_config, methods)
    result = {
        "source_sha256": source_hash(root),
        "dataset_id": manifest["dataset_id"],
        "environment": environment(),
        "measured": False,
        "passes_budget": False,
        "target_loader_accessed": False,
        "methods": {},
        "optimizer_steps_per_method": steps,
        "status": "running",
    }
    atomic_json(destination, result)
    if base_config.device != "cuda" or not torch.cuda.is_available():
        result["reason"] = "CUDA unavailable; CPU checks cannot establish GPU fit"
        result["status"] = "unavailable"
        atomic_json(destination, result)
        return result
    if steps < 10:
        raise ValueError("At least 10 accepted real-data CUDA updates are required")
    for method in methods:
        print(
            f"[preflight] {base_config.dataset} {method}: {steps} GPU updates "
            f"activation={base_config.activation} "
            f"train_batch={base_config.batch_size} eval_batch={base_config.eval_batch_size} "
            f"input={base_config.input_shape[0]}x{base_config.input_shape[1]} "
            f"train_precision={base_config.precision} eval_precision={base_config.eval_precision}",
            flush=True,
        )
        config = replace(base_config, method=method).validate()
        result["active"] = {
            "dataset": config.dataset,
            "method": method,
            "stage": "setup",
            "accepted_optimizer_steps": 0,
            "config": config.to_dict(),
        }
        atomic_json(destination, result)
        try:
            result["methods"][method] = profile_method(manifest, config, steps, result["active"])
        except Exception as error:
            result["status"] = "failed"
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            atomic_json(destination, result)
            raise RuntimeError(
                f"Preflight failed: dataset={config.dataset} method={method} "
                f"stage={result['active']['stage']} "
                f"accepted_updates={result['active']['accepted_optimizer_steps']}/{steps}; "
                f"{type(error).__name__}: {error}. Diagnostics: {destination}"
            ) from error
        statistics = result["methods"][method]["vram"]
        result["active"]["stage"] = "complete"
        atomic_json(destination, result)
        print(
            f"[preflight] {config.dataset} {method}: "
            f"train={result['methods'][method]['training_images_per_second']:.1f} images/s "
            f"eval={result['methods'][method]['evaluation_images_per_second']:.1f} images/s "
            f"peak_reserved={statistics['peak_reserved_gib']:.2f} GiB",
            flush=True,
        )
    result["measured"] = True
    result["status"] = "complete"
    result["environment"] = environment()
    result["matched_initial_weights"] = (
        len({item["initial_shared_model_sha256"] for item in result["methods"].values()}) == 1
    )
    result["matched_augmented_batches"] = (
        len({item["first_augmented_batch_sha256"] for item in result["methods"].values()}) == 1
    )
    result["passes_budget"] = (
        result["matched_initial_weights"]
        and result["matched_augmented_batches"]
        and all(
            item["backbone_updated"]
            and item["vram"]["peak_reserved_gib"] <= base_config.max_vram_gib
            for item in result["methods"].values()
        )
    )
    atomic_json(destination, result)
    return result


def verify_profile(value, root, manifest, base_config, methods=None):
    """Reject unmeasured, stale, changed-device, or reduced-batch CUDA preflight evidence."""
    methods = selected_methods(base_config, methods)
    if (
        not value.get("measured")
        or not value.get("passes_budget")
        or value.get("source_sha256") != source_hash(root)
        or value.get("dataset_id") != manifest["dataset_id"]
    ):
        raise ValueError("A matching successful real-data CUDA preflight is required")
    if value["environment"] != environment() or value.get("optimizer_steps_per_method", 0) < 10:
        raise ValueError("CUDA preflight environment or step budget differs")
    if set(value["methods"]) != set(methods):
        raise ValueError("CUDA preflight omitted an intervention")
    for method in methods:
        signature = memory_signature(replace(base_config, method=method))
        if value["methods"][method]["signature"] != signature:
            raise ValueError("CUDA preflight shape/precision signature differs")
