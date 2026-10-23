"""Measure real full-batch training and evaluation before permitting the main study."""

import gc
import hashlib
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import torch

from cue_routing.engine import validate_device, vram_stats
from cue_routing.io import atomic_json, canonical_hash, environment, seed_everything

from .config import MAIN_METHODS, MECHANISM_METHODS
from .data import make_loader
from .engine import evaluate_loader, make_model, optimize_batch, optimizer_for
from .identity import source_hash


def memory_signature(config):
    """Ignore scalar loss coefficients while binding all memory-relevant recipe fields."""
    return {
        key: getattr(config, key)
        for key in (
            "dataset",
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
    """Measure one intervention and retain its failure stage without retaining live workers."""
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
    loader = make_loader(manifest, config, "train", epoch=1)
    first_batch = None
    model.train()
    activity["stage"] = "training"
    with ExitStack() as stack:
        iterator = stack.enter_context(profile_iterator(loader))
        began = time.monotonic()
        for index in range(steps):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = stack.enter_context(profile_iterator(loader))
                images, labels = next(iterator)
            if len(images) != config.batch_size:
                raise ValueError("Preflight requires the full declared training batch")
            if index == 0:
                first_batch = canonical_hash(
                    {
                        "images": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                        "labels": labels.tolist(),
                    }
                )
            diagnostics = optimize_batch(
                model, optimizer, scaler, images.to(config.device), labels.to(config.device), config
            )
            activity["accepted_optimizer_steps"] = index + 1
        torch.cuda.synchronize()
        seconds = time.monotonic() - began
    activity["stage"] = "validation"
    evaluation_began = time.monotonic()
    with profile_iterator(make_loader(manifest, config, "validation")) as iterator:
        validation_batch = next(iterator)
        if len(validation_batch[0]) != config.eval_batch_size:
            raise ValueError("Preflight requires the full declared evaluation batch")
        evaluate_loader(model, [validation_batch], config)
        torch.cuda.synchronize()
        evaluation_seconds = time.monotonic() - evaluation_began
    return {
        "signature": memory_signature(config),
        "initial_shared_model_sha256": initial,
        "first_augmented_batch_sha256": first_batch,
        "accepted_optimizer_steps": steps,
        "backbone_updated": not torch.equal(before, next(model.parameters()).detach()),
        "training_images_per_second": steps * config.batch_size / max(seconds, 1e-9),
        "evaluation_images_per_second": len(validation_batch[0]) / max(evaluation_seconds, 1e-9),
        "vram": vram_stats(config),
        "last_diagnostics": diagnostics,
        "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def profile_dataset(root, manifest, base_config, destination, steps=10):
    """Use actual train/validation images, SGD momentum, AMP, and every selected intervention."""
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
    methods = MECHANISM_METHODS if base_config.dataset == "dominoes" else MAIN_METHODS
    for method in methods:
        print(
            f"[preflight] {base_config.dataset} {method}: {steps} GPU updates "
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


def verify_profile(value, root, manifest, base_config):
    """Reject unmeasured, stale, changed-device, or reduced-batch CUDA preflight evidence."""
    methods = MECHANISM_METHODS if base_config.dataset == "dominoes" else MAIN_METHODS
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
