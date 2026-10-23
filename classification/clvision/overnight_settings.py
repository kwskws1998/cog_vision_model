"""One-seed CIFAR comparison with fixed common settings and a finite CUE-only search."""

import math
from datetime import datetime

from .config import Config

CUE_FIELDS = {"gate_tau", "gate_temperature", "joint_floor"}


def deadline_time(value):
    """Require an explicit timezone so a remote server's local clock cannot shift the deadline."""
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("Deadline must include its timezone, for example +09:00")
    return result.timestamp()


def validate_settings(value):
    """Forbid dataset/seed sweeps or optimizer changes disguised as CUE tuning."""
    if set(value) != {"training", "candidates", "reserve_minutes", "time_safety_factor"}:
        raise ValueError("Unknown or missing overnight settings")
    config = Config.from_dict(value["training"])
    if config.dataset != "cifar100" or config.activation != "preact" or config.method != "erm":
        raise ValueError("This comparison requires CIFAR-100, PreAct ResNet-18 and an ERM base")
    if not config.debug and (
        config.epochs != 200
        or config.batch_size != 128
        or config.eval_batch_size != 128
        or config.learning_rate != 0.1
        or config.warmup_epochs != 0
        or config.milestones != (100, 150)
        or config.lr_decay != 0.1
        or config.momentum != 0.9
        or config.weight_decay != 1e-4
        or config.precision != "fp32"
        or config.eval_precision != "fp32"
        or config.initialization != "msra"
        or config.normalization != "zipfls"
        or config.gradient_clip != 0
    ):
        raise ValueError(
            "Common paper recipe is fixed; only seed, workers and VRAM ceiling may vary"
        )
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Register at least one CUE candidate")
    resolved = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != CUE_FIELDS:
            raise ValueError("Candidates must specify only gate_tau, gate_temperature, joint_floor")
        item = Config.from_dict({**config.to_dict(), **candidate, "method": "cue"})
        if item in resolved:
            raise ValueError("Duplicate CUE candidate")
        resolved.append(item)
    for key, minimum in (("reserve_minutes", 0), ("time_safety_factor", 1)):
        item = value[key]
        if type(item) not in (int, float) or not math.isfinite(item) or item < minimum:
            raise ValueError(f"Invalid {key}")
    return config, resolved


def preview(value, finish_by):
    """Describe actual splits, the single training seed, and unavoidable reproduction differences."""
    config, candidates = validate_settings(value)
    deadline_time(finish_by)
    return {
        "mode": "one_seed_validation_selected_cue",
        "seed": config.seed,
        "dataset": "cifar100",
        "splits": {"train": 45000, "validation": 5000, "test": 10000},
        "base_config": config.to_dict(),
        "cue_candidates": [item.to_dict() for item in candidates],
        "maximum_training_runs": 1 + len(candidates),
        "finish_by": finish_by,
        "reserve_minutes": value["reserve_minutes"],
        "time_safety_factor": value["time_safety_factor"],
        "selection": "completed final-epoch validation accuracy, then NLL, then registered order",
        "test_policy": "only ERM and selected CUE, after immutable search closure",
        "winner_policy": "reuse the selected trained checkpoint; no second seed or retraining",
        "deadline_policy": "skip not-yet-started candidates using conservative measured time; never shorten epochs",
        "exact_published_score_reproduction": False,
        "differences": [
            "45k train / 5k validation instead of training on all 50k",
            "MSRA Conv fan-out initialization follows paper prose; public CIFAR constructor uses defaults",
            "LR reduced after epochs 100/150, not the public-code off-by-one update",
            "keyed augmentation and sample order, deterministic kernels, one explicit seed",
            "official test is sealed until all search decisions are frozen",
        ],
    }
