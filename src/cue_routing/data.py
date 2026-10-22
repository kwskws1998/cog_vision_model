"""Pinned PACS splits, content manifests, and stateless balanced augmentation."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from .config import CLASSES, DOMAINS
from .io import atomic_json, canonical_hash, file_hash, read_json

EXPECTED = {
    "art_painting": (1840, 208, 2048),
    "cartoon": (2107, 237, 2344),
    "photo": (1499, 171, 1670),
    "sketch": (3531, 398, 3929),
}
PACS_URL = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
DASSL_PACS_URL = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"


def safe_relative(value):
    """Reject absolute paths, traversal, backslashes, and malformed PACS paths."""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or len(path.parts) != 3
        or path.as_posix() != value
    ):
        raise ValueError(f"Unsafe PACS path: {value}")
    if path.parts[0] not in DOMAINS or path.parts[1] not in CLASSES:
        raise ValueError(f"Unknown PACS domain or class: {value}")
    return path.as_posix()


def checked_image_path(root, name):
    """Reject symbolic links in any image path component, including class directories."""
    root = Path(root).resolve()
    relative = PurePosixPath(safe_relative(name))
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinked dataset path: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise FileNotFoundError(f"Missing or unsafe image: {name}")
    return path


def read_official_splits(reference_dir):
    """Validate official 1-based labels, exact counts, and disjoint train/val sets."""
    splits, hashes = {}, {}
    provenance = read_json(Path(reference_dir).parent / "provenance.json")
    for domain in DOMAINS:
        splits[domain] = {}
        for split, expected in zip(("train", "crossval", "test"), EXPECTED[domain]):
            path = Path(reference_dir) / f"{domain}_{split}_kfold.txt"
            content = path.read_bytes()
            blob_hash = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if blob_hash != provenance["split_git_blob_sha1"][path.name]:
                raise ValueError(f"Split differs from the pinned GitHub blob: {path.name}")
            records = []
            for line in path.read_text().splitlines():
                name, label = line.rsplit(maxsplit=1)
                name, label = safe_relative(name), int(label) - 1
                if (
                    not 0 <= label < len(CLASSES)
                    or name.split("/")[0] != domain
                    or CLASSES[label] != name.split("/")[1]
                ):
                    raise ValueError("Split domain/class does not match its label")
                records.append([name, label])
            if len(records) != expected or len({item[0] for item in records}) != expected:
                raise ValueError(f"Wrong or duplicated split count: {path}")
            splits[domain][split] = records
            hashes[path.name] = file_hash(path)
        train, valid, test = (dict(splits[domain][key]) for key in ("train", "crossval", "test"))
        if set(train) & set(valid) or {**train, **valid} != test:
            raise ValueError("Official train/validation partition is inconsistent")
    return splits, hashes


def prepare_manifest(data_root, manifest_path, reference_dir):
    """Verify every real image once and fingerprint the complete benchmark package."""
    data_root = Path(data_root).resolve()
    splits, split_hashes = read_official_splits(reference_dir)
    images = {}
    for domain in DOMAINS:
        for name, _ in splits[domain]["test"]:
            path = checked_image_path(data_root, name)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
            images[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    content = {
        "schema": 1,
        "kind": "pacs",
        "classes": list(CLASSES),
        "splits": splits,
        "split_hashes": split_hashes,
        "images": images,
    }
    manifest = {**content, "dataset_id": canonical_hash(content), "data_root": str(data_root)}
    atomic_json(manifest_path, manifest)
    return manifest


def load_manifest(path, reference_dir, verify_images=True):
    """Recheck dataset integrity before using a manifest as experiment provenance."""
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    content = {
        key: value for key, value in manifest.items() if key not in ("dataset_id", "data_root")
    }
    if canonical_hash(content) != manifest["dataset_id"] or manifest["classes"] != list(CLASSES):
        raise ValueError("Manifest checksum or class mapping changed")
    if manifest["kind"] == "pacs":
        splits, hashes = read_official_splits(reference_dir)
        if manifest["splits"] != splits or manifest["split_hashes"] != hashes:
            raise ValueError("Manifest does not use the pinned official PACS splits")
    elif manifest["kind"] != "synthetic_fixture":
        raise ValueError("Unknown dataset kind")
    expected_names = set()
    for domain in DOMAINS:
        split = manifest["splits"][domain]
        training, validation, testing = (dict(split[key]) for key in ("train", "crossval", "test"))
        if set(training) & set(validation) or {**training, **validation} != testing:
            raise ValueError("Train/validation split overlap or incomplete partition")
        for name, label in testing.items():
            safe_relative(name)
            if not 0 <= label < len(CLASSES) or name.split("/")[:2] != [domain, CLASSES[label]]:
                raise ValueError("Corrupt split label")
        expected_names.update(testing)
    if expected_names != set(manifest["images"]):
        raise ValueError("Manifest image set differs from the split union")
    if verify_images:
        root = Path(manifest["data_root"])
        for name, expected in manifest["images"].items():
            image_path = checked_image_path(root, name)
            if (
                image_path.stat().st_size != expected["bytes"]
                or file_hash(image_path) != expected["sha256"]
            ):
                raise ValueError(f"Image content changed or is missing: {name}")
    return manifest


def extract_archive(archive, destination):
    """Safely import a PACS ZIP without overwriting an existing dataset directory."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            if sum(item.file_size for item in handle.infolist()) > 5 * 2**30:
                raise ValueError("Unexpectedly large PACS archive")
            for item in handle.infolist():
                name = PurePosixPath(item.filename)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)
                ):
                    raise ValueError("Unsafe archive member")
            handle.extractall(temporary)
        candidates = [temporary] + [
            item for pattern in ("*", "*/*") for item in temporary.glob(pattern) if item.is_dir()
        ]
        roots = [item for item in candidates if all((item / domain).is_dir() for domain in DOMAINS)]
        if len(roots) != 1:
            raise ValueError("Cannot identify exactly one PACS image root")
        shutil.move(str(roots[0]), str(destination))
    return {"archive_sha256": file_hash(archive), "archive_path": str(Path(archive).resolve())}


def download_pacs(path):
    """Try documented DomainBed/Dassl archives and report the actual successful source."""
    import gdown

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists():
        raise FileExistsError(path)
    failures = []
    for url in (PACS_URL, DASSL_PACS_URL):
        try:
            result = gdown.download(url, str(temporary), quiet=False, use_cookies=False)
            if result is None or not zipfile.is_zipfile(temporary):
                raise RuntimeError("Server did not provide a ZIP archive")
            os.replace(temporary, path)
            return {"source_url": url, "earlier_download_failures": failures}
        except Exception as error:
            temporary.unlink(missing_ok=True)
            failures.append({"url": url, "error": f"{type(error).__name__}: {error}"})
    atomic_json(path.with_suffix(".download_failure.json"), {"attempts": failures})
    raise RuntimeError(
        "Both documented PACS links failed. Import an existing PACS ZIP with prepare --archive, or point prepare --data-root at extracted images. Download errors were saved beside the requested archive."
    )


def image_transform(size, training):
    """Use one identical ImageNet-normalized augmentation recipe for all arms."""
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(round(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class PACSDataset(Dataset):
    """Read only the explicitly requested domains and split; never infer a split."""

    def __init__(self, manifest, domains, split, size, training=False):
        if (
            split not in ("train", "crossval", "test")
            or not domains
            or len(set(domains)) != len(domains)
        ):
            raise ValueError("Invalid dataset partition")
        self.root = Path(manifest["data_root"])
        self.records = []
        self.domain_indices = {}
        for domain in domains:
            begin = len(self.records)
            self.records.extend(manifest["splits"][domain][split])
            self.domain_indices[domain] = list(range(begin, len(self.records)))
        self.transform = image_transform(size, training)
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        """Isolate augmentation RNG from the model and worker scheduling."""
        if isinstance(key, tuple):
            index, augmentation_seed = key
        else:
            index, augmentation_seed = key, None
        if self.training and augmentation_seed is None:
            raise ValueError("Training requires a stateless augmentation seed")
        name, label = self.records[index]
        with Image.open(self.root / name) as image:
            image = image.convert("RGB")
            with torch.random.fork_rng(devices=[]):
                if augmentation_seed is not None:
                    torch.random.default_generator.manual_seed(augmentation_seed)
                tensor = self.transform(image)
        return tensor, label


def keyed_seed(*values):
    """Derive stable independent seeds without Python's randomized hash function."""
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little"
    ) % (2**63 - 1)


class BalancedStepSampler(Sampler):
    """Sample each source domain with replacement using only (seed, step, domain)."""

    def __init__(self, domain_indices, per_domain, seed, start_step, end_step):
        if (
            per_domain < 1
            or start_step < 0
            or end_step < start_step
            or any(not indices for indices in domain_indices.values())
        ):
            raise ValueError("Invalid balanced step sampler")
        self.domain_indices = domain_indices
        self.per_domain, self.seed, self.start_step, self.end_step = (
            per_domain,
            seed,
            start_step,
            end_step,
        )

    def __len__(self):
        return self.end_step - self.start_step

    def __iter__(self):
        """Make resume and worker-count changes preserve exact images and augments."""
        for step in range(self.start_step, self.end_step):
            batch = []
            for domain, indices in self.domain_indices.items():
                generator = torch.Generator().manual_seed(
                    keyed_seed(self.seed, step, domain, "sample")
                )
                choices = torch.randint(
                    len(indices), (self.per_domain,), generator=generator
                ).tolist()
                batch.extend(
                    (indices[index], keyed_seed(self.seed, step, domain, slot, "augment"))
                    for slot, index in enumerate(choices)
                )
            yield batch


def training_loader(manifest, config, start_step=0, end_step=None):
    """Construct a source-only balanced loader with an isolated worker RNG."""
    domains = [domain for domain in DOMAINS if domain != config.target]
    dataset = PACSDataset(manifest, domains, "train", config.image_size, training=True)
    sampler = BalancedStepSampler(
        dataset.domain_indices,
        config.batch_per_domain,
        config.seed,
        start_step,
        config.steps if end_step is None else end_step,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.workers,
        pin_memory=config.device == "cuda",
        generator=torch.Generator().manual_seed(keyed_seed(config.seed, "loader")),
    )


def evaluation_loader(manifest, domain, split, config):
    """Create a deterministic unaugmented loader for one explicitly named partition."""
    dataset = PACSDataset(manifest, [domain], split, config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.device == "cuda",
        generator=torch.Generator().manual_seed(0),
    )


def make_fixture(directory, reference_dir):
    """Create tiny artificial images solely for pipeline verification, never accuracy claims."""
    import numpy as np

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    splits, images = {}, {}
    generator = np.random.default_rng(173)
    for domain in DOMAINS:
        splits[domain] = {"train": [], "crossval": [], "test": []}
        for label, category in enumerate(CLASSES):
            for index in range(3):
                name = f"{domain}/{category}/fixture_{index}.png"
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(generator.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)
                record = [name, label]
                splits[domain]["train" if index < 2 else "crossval"].append(record)
                splits[domain]["test"].append(record)
                images[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    content = {
        "schema": 1,
        "kind": "synthetic_fixture",
        "classes": list(CLASSES),
        "splits": splits,
        "split_hashes": {},
        "images": images,
    }
    manifest = {**content, "dataset_id": canonical_hash(content), "data_root": str(directory)}
    path = directory / "manifest.json"
    atomic_json(path, manifest)
    return path
