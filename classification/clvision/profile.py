"""Measure real full-batch training and evaluation before permitting the main study."""

import gc
import hashlib
import time
from dataclasses import replace

import torch

from cue_routing.engine import vram_stats
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
            "max_vram_gib",
            "debug",
        )
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
    }
    atomic_json(destination, result)
    if base_config.device != "cuda" or not torch.cuda.is_available():
        result["reason"] = "CUDA unavailable; CPU checks cannot establish GPU fit"
        atomic_json(destination, result)
        return result
    if steps < 10:
        raise ValueError("At least10 accepted real-data CUDA updates are required")
    methods = MECHANISM_METHODS if base_config.dataset == "dominoes" else MAIN_METHODS
    for method in methods:
        config = replace(base_config, method=method).validate()
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
        iterator = iter(loader)
        began = time.monotonic()
        model.train()
        diagnostics = []
        for index in range(steps):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
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
            diagnostics.append(
                optimize_batch(
                    model, optimizer, scaler, images.to("cuda"), labels.to("cuda"), config
                )
            )
        torch.cuda.synchronize()
        seconds = time.monotonic() - began
        validation_iterator = iter(make_loader(manifest, config, "validation"))
        validation_batch = next(validation_iterator)
        if len(validation_batch[0]) != config.eval_batch_size:
            raise ValueError("Preflight requires the full declared evaluation batch")
        evaluate_loader(model, [validation_batch], config)
        statistics = vram_stats(config)
        result["methods"][method] = {
            "signature": memory_signature(config),
            "initial_shared_model_sha256": initial,
            "first_augmented_batch_sha256": first_batch,
            "accepted_optimizer_steps": steps,
            "backbone_updated": not torch.equal(before, next(model.parameters()).detach()),
            "training_images_per_second": steps * config.batch_size / max(seconds, 1e-9),
            "vram": statistics,
            "last_diagnostics": diagnostics[-1],
            "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        del (
            model,
            optimizer,
            scaler,
            images,
            labels,
            before,
            loader,
            iterator,
            validation_iterator,
            validation_batch,
        )
    result["measured"] = True
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
