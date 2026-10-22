"""Validated, serialized experiment contracts shared by all comparison arms."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
METHODS = ("erm", "rsc", "cue")


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "cue"
    target: str = "sketch"
    seed: int = 0
    steps: int = 5000
    eval_every: int = 200
    batch_per_domain: int = 16
    image_size: int = 224
    eval_batch_size: int = 64
    learning_rate: float = 0.00005
    weight_decay: float = 0.0
    precision: str = "fp16"
    device: str = "cuda"
    workers: int = 4
    freeze_bn: bool = True
    pretrained: bool = True
    pretrained_path: str = ""
    grid_size: int = 2
    gate_tau: float = 0.15
    gate_temperature: float = 0.05
    joint_floor: float = 0.5
    routing: str = "cue"
    fixed_joint: float = 0.75
    rsc_feature_drop: float = 1.0 / 3.0
    rsc_batch_drop: float = 1.0 / 3.0
    gradient_clip: float = 0.0
    max_vram_gib: float = 20.0
    debug: bool = False

    def validate(self):
        """Reject unsupported or ambiguous settings before any data are accessed."""
        for name in (
            "steps",
            "eval_every",
            "batch_per_domain",
            "image_size",
            "eval_batch_size",
            "grid_size",
            "seed",
            "workers",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("freeze_bn", "pretrained", "debug"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "learning_rate",
            "weight_decay",
            "gate_tau",
            "gate_temperature",
            "joint_floor",
            "fixed_joint",
            "rsc_feature_drop",
            "rsc_batch_drop",
            "gradient_clip",
            "max_vram_gib",
        ):
            if type(getattr(self, name)) not in (float, int) or not math.isfinite(
                getattr(self, name)
            ):
                raise ValueError(f"{name} must be finite")
        for name in ("method", "target", "precision", "device", "routing", "pretrained_path"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if self.method not in METHODS or self.target not in DOMAINS:
            raise ValueError("Unknown method or target domain")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Only explicit cpu and cuda devices are supported")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU verification uses fp32")
        if self.routing not in ("cue", "image", "fixed", "shuffled"):
            raise ValueError("Unknown cue-routing ablation")
        for name in (
            "steps",
            "eval_every",
            "batch_per_domain",
            "image_size",
            "eval_batch_size",
            "grid_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.grid_size > (self.image_size + 31) // 32 or self.workers < 0:
            raise ValueError("Invalid feature-grid size or worker count")
        for name in ("joint_floor", "fixed_joint"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.gate_tau <= 1.0 or self.gate_temperature <= 0.0:
            raise ValueError("Invalid gate threshold or temperature")
        if not 0.0 < self.rsc_feature_drop < 1.0 or not 0.0 < self.rsc_batch_drop < 1.0:
            raise ValueError("RSC drop fractions must be strictly between zero and one")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip < 0
            or self.max_vram_gib <= 0
        ):
            raise ValueError("Invalid optimization or memory setting")
        if not self.debug and (self.image_size != 224 or self.steps < 5000 or not self.pretrained):
            raise ValueError("Short, resized, or randomly initialized runs require debug=true")
        return self

    def to_dict(self):
        """Return a plain serialization with no device-specific Python objects."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        """Fail on misspelled keys rather than silently discarding settings."""
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        """Read a complete or default-overriding JSON configuration."""
        return cls.from_dict(json.loads(Path(path).read_text()))
