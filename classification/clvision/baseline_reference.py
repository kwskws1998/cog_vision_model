"""Audited CIFAR-100 CE reference protocol, independent of the CUE study's held-out split."""

import importlib.machinery
import importlib.util
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cue_routing.io import file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "classification/references/zipfls"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


@dataclass
class BaselineConfig:
    """Expose experiment inputs while fixing the backbone, CE-only objective and full split."""

    epochs: int = 200
    batch_size: int = 128
    eval_batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    milestones: list = field(default_factory=lambda: [100, 150])
    lr_decay: float = 0.1
    schedule: str = "official-code"
    seeds: list = field(default_factory=lambda: [0])
    workers: int = 4
    device: str = "cuda"

    def validate(self):
        """Reject ambiguous or invalid recipes before loading data or allocating CUDA memory."""
        for key in ("epochs", "batch_size", "eval_batch_size"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        for key in ("learning_rate", "momentum", "weight_decay", "lr_decay"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.learning_rate <= 0 or not 0 < self.lr_decay < 1:
            raise ValueError("learning_rate must be positive and lr_decay in (0, 1)")
        if not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("momentum must be in [0, 1) and weight_decay nonnegative")
        if (
            not isinstance(self.seeds, list)
            or not self.seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be a nonempty list of unique integers in [0, 2**32)")
        if (
            not isinstance(self.milestones, list)
            or any(type(e) is not int or e < 1 for e in self.milestones)
            or self.milestones != sorted(set(self.milestones))
        ):
            raise ValueError("milestones must be sorted unique positive integers")
        if self.schedule not in ("official-code", "paper"):
            raise ValueError("schedule must be official-code or paper")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        return self

    def rate(self, epoch):
        """Use one-based epochs; preserve the author's end-of-zero-based-epoch LR update."""
        offset = 2 if self.schedule == "official-code" else 1
        drops = sum(epoch >= milestone + offset for milestone in self.milestones)
        return self.learning_rate * self.lr_decay**drops

    def preview(self):
        """Describe the recipe and distinguish code fidelity from paper prose and tuning runs."""
        return {
            "config": asdict(self),
            "dataset": "CIFAR100",
            "input_size": [32, 32],
            "training_examples": 50000,
            "internal_validation_examples": 0,
            "evaluation_examples": 10000,
            "evaluation_split": "official_test",
            "model": "author CIFAR_ResNet18(PreActBlock, [2,2,2,2])",
            "objective": "final-head cross entropy only; no auxiliary CE or distillation",
            "initialization": "author CIFAR constructor defaults; no added MSRA reinitialization",
            "precision": "fp32",
            "tf32": False,
            "warmup_epochs": 0,
            "nesterov": False,
            "gradient_accumulation_steps": 1,
            "normalization": {"mean": list(MEAN), "std": list(STD)},
            "lr_drop_first_used_epochs": [
                e + (2 if self.schedule == "official-code" else 1) for e in self.milestones
            ],
            "evaluation_policy": "official_test each epoch; report final and best-observed separately",
            "tuning": False,
            "published_cifar100_reference_percent": {"mean": 75.51, "sd": 0.28, "runs": 5},
            "score_reproduction_verified": False,
        }


def checked_reference():
    """Check the pinned author files before executing the unmodified backbone source."""
    provenance = read_json(ROOT / "classification/references/zipfls_provenance.json")
    for item in provenance["files"]:
        if file_hash(ROOT / item["file"]) != item["sha256"]:
            raise ValueError(f"Pinned reference hash mismatch: {item['file']}")
    return provenance


def build_model():
    """Execute the pinned constructor, preserving unused heads and its complete RNG consumption."""
    checked_reference()
    path = REFERENCE / "pytorch_zipfls__cifar_models__resnet.py.txt"
    loader = importlib.machinery.SourceFileLoader("eccv2022_cifar_reference", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.CIFAR_ResNet18(num_classes=100, upsample="bilinear")


def image_transform(train):
    """Match the author's CIFAR transform order, zero padding, flip and normalization constants."""
    spatial = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(
        (spatial if train else [transforms.Resize(32)])
        + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def load_data(data_root, download=False):
    """Reuse checked raw CIFAR files and train on all 50k examples, without creating study splits."""
    root = Path(data_root).resolve()
    candidates = [root / "raw/cifar100", root / "raw/cifar100/checked", root]
    raw = next(
        (p for p in candidates if (p / datasets.CIFAR100.base_folder).is_dir()),
        root / "raw/cifar100",
    )
    train = datasets.CIFAR100(raw, train=True, transform=image_transform(True), download=download)
    test = datasets.CIFAR100(raw, train=False, transform=image_transform(False), download=download)
    for data, count in ((train, 50000), (test, 10000)):
        if data.data.shape != (count, 32, 32, 3):
            raise ValueError("Unexpected official CIFAR-100 shape")
        if np.bincount(data.targets, minlength=100).tolist() != [count // 100] * 100:
            raise ValueError("Unexpected official CIFAR-100 class counts")
    files = [name for name, _ in datasets.CIFAR100.train_list + datasets.CIFAR100.test_list]
    files.append(datasets.CIFAR100.meta["filename"])
    identity = {
        "kind": "official_cifar100",
        "train_n": len(train),
        "test_n": len(test),
        "sha256": {name: file_hash(raw / datasets.CIFAR100.base_folder / name) for name in files},
    }
    return train, test, identity


def make_loaders(train, test, config):
    """Preserve shuffled full training batches, the short last batch and worker restart semantics."""
    return (
        DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=config.device == "cuda",
            drop_last=False,
        ),
        DataLoader(
            test,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=config.device == "cuda",
            drop_last=False,
        ),
    )
