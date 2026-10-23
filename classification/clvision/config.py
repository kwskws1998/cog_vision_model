"""Explicit recipes for natural-image comparisons and the controlled cue task."""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

MAIN_METHODS = ("erm", "label_smoothing", "fixed_global_local", "zipfls_dense2", "cue")
MECHANISM_METHODS = (
    "erm",
    "fixed_global_local",
    "cue",
    "pure_marginal",
    "image_mean_gate",
    "permuted_gate",
)
DATASETS = {"cifar100": (100, 32, 32), "tiny_imagenet": (200, 64, 64), "dominoes": (2, 64, 32)}
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


@dataclass(frozen=True)
class Config:
    """Validate every public setting before creating a run or a data loader."""

    dataset: str = "cifar100"
    method: str = "cue"
    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    milestones: tuple = (100, 150)
    lr_decay: float = 0.1
    gate_tau: float = 0.15
    gate_temperature: float = 0.05
    joint_floor: float = 0.5
    fixed_lambda: float = 0.25
    zipf_lambda: float = 0.1
    zipf_aux_weight: float = 0.1
    zipf_power: float = 1.0
    label_smoothing: float = 0.1
    workers: int = 4
    device: str = "cuda"
    precision: str = "fp16"
    max_vram_gib: float = 20.0
    gradient_clip: float = 0.0
    debug: bool = False
    debug_max_batches: int = 0

    def validate(self):
        """Reject unsafe numeric or device combinations before a recipe is frozen."""
        if self.dataset not in DATASETS or self.method not in set(MAIN_METHODS + MECHANISM_METHODS):
            raise ValueError("Unknown dataset or comparison method")
        for key in (
            "seed",
            "epochs",
            "batch_size",
            "eval_batch_size",
            "workers",
            "debug_max_batches",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < (
                1 if key in ("epochs", "batch_size", "eval_batch_size") else 0
            ):
                raise ValueError(f"Invalid integer {key}")
        if self.seed >= 2**63 or type(self.debug) is not bool:
            raise ValueError("Invalid seed or debug flag")
        if self.batch_size < 2:
            raise ValueError("Training BatchNorm requires a batch size of at least two")
        if (
            not isinstance(self.milestones, tuple)
            or any(
                type(epoch) is not int or not 0 < epoch < self.epochs for epoch in self.milestones
            )
            or tuple(sorted(set(self.milestones))) != self.milestones
        ):
            raise ValueError("Milestones must be unique, sorted, and within the training schedule")
        for key in (
            "learning_rate",
            "momentum",
            "weight_decay",
            "lr_decay",
            "gate_tau",
            "gate_temperature",
            "joint_floor",
            "fixed_lambda",
            "zipf_lambda",
            "zipf_aux_weight",
            "zipf_power",
            "label_smoothing",
            "max_vram_gib",
            "gradient_clip",
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or nonnumeric {key}")
        if not (self.learning_rate > 0 and 0 <= self.momentum < 1 and self.weight_decay >= 0):
            raise ValueError("Invalid optimizer settings")
        if not (0 < self.lr_decay <= 1 and self.gate_temperature > 0 and 0 <= self.gate_tau <= 1):
            raise ValueError("Invalid scheduler or gate settings")
        if not (0 <= self.joint_floor <= 1 and 0 <= self.fixed_lambda <= 1):
            raise ValueError("Invalid consistency coefficients")
        if not (self.zipf_lambda >= 0 and self.zipf_aux_weight >= 0 and self.zipf_power > 0):
            raise ValueError("Invalid Zipf coefficients")
        if not (
            0 <= self.label_smoothing < 1
            and 0 < self.max_vram_gib <= 24
            and self.gradient_clip >= 0
        ):
            raise ValueError("Invalid smoothing, clipping, or memory settings")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unsupported device or precision")
        if self.device == "cpu" and (self.precision != "fp32" or not self.debug):
            raise ValueError("CPU is only supported as explicit FP32 debug verification")
        if self.debug_max_batches and not self.debug:
            raise ValueError("Reduced training requires explicit debug=true")
        return self

    def to_dict(self):
        """Use portable JSON types in all contracts."""
        value = asdict(self)
        value["milestones"] = list(self.milestones)
        return value

    @classmethod
    def from_dict(cls, value):
        """Reject unknown fields rather than silently dropping stale settings."""
        value = dict(value)
        if "milestones" in value:
            value["milestones"] = tuple(value["milestones"])
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        """Read and validate a saved experiment configuration."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def recipe(dataset, **overrides):
    """Construct one dataset's declared recipe with explicit development overrides."""
    value = Config(dataset=dataset)
    if dataset == "tiny_imagenet":
        value = replace(value, zipf_lambda=1.0, zipf_aux_weight=0.5)
    if dataset == "dominoes":
        value = replace(value, epochs=50, milestones=(25, 40))
    return replace(value, **overrides).validate()
