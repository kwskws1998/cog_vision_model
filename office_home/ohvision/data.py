"""Verified Office-Home packages, frozen stratified splits, and balanced source loaders."""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cue_routing.data import BalancedStepSampler, image_transform, keyed_seed
from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json

from .config import CLASSES, DOMAINS, EXPECTED_COUNTS, SPLIT_POLICY

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_ARCHIVE_BYTES = 8 * 2**30


def folder_key(value):
    """Resolve documented case/underscore/space variations without changing class meaning."""
    return re.sub(r"[ _-]", "", value).casefold()


def canonical_name(value):
    """Map flat or Dassl train/val paths to one canonical domain/class/image identity."""
    if not isinstance(value, str):
        raise ValueError("Image paths must be strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or len(path.parts) not in (3, 4)
        or any(part.startswith(".") for part in path.parts)
        or path.suffix.casefold() not in IMAGE_SUFFIXES
    ):
        raise ValueError(f"Unsafe Office-Home image path: {value}")
    if len(path.parts) == 4 and path.parts[1] not in ("train", "val"):
        raise ValueError("Only explicitly identified Dassl train/val layouts are supported")
    domains = {folder_key(name): name for name in DOMAINS}
    classes = {folder_key(name): name for name in CLASSES}
    try:
        domain, category = domains[folder_key(path.parts[0])], classes[folder_key(path.parts[-2])]
    except KeyError as error:
        raise ValueError(f"Unknown Office-Home domain or class: {value}") from error
    return f"{domain}/{category}/{path.name}"


def safe_relative(value):
    """Require the canonical three-component image identity used by every split."""
    if canonical_name(value) != value or len(PurePosixPath(value).parts) != 3:
        raise ValueError(f"Noncanonical Office-Home image identity: {value}")
    return value


def checked_image_path(root, relative):
    """Check each physical path component, rejecting links and traversal before image access."""
    canonical_name(relative)
    root = Path(root).resolve()
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinked Office-Home path: {relative}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise FileNotFoundError(f"Missing Office-Home image: {relative}")
    return path


def discover_images(data_root):
    """Index every image and reject ambiguous aliases, nested files, and duplicate identities."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    found, directory_names = {}, {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinked dataset member: {path}")
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        identity = canonical_name(relative)
        if identity in found:
            raise ValueError(f"Duplicated image identity across physical partitions: {identity}")
        canonical_domain, canonical_class, _ = identity.split("/")
        physical = PurePosixPath(relative).parts
        for key, actual in (
            ((canonical_domain,), physical[0]),
            ((canonical_domain, canonical_class), physical[-2]),
        ):
            if key in directory_names and directory_names[key] != actual:
                raise ValueError("Conflicting aliases for one Office-Home domain or class")
            directory_names[key] = actual
        found[identity] = relative
    return dict(sorted(found.items()))


def build_splits(names):
    """Freeze a label-stratified 80/20 source partition independently of training seeds."""
    grouped = {domain: {category: [] for category in CLASSES} for domain in DOMAINS}
    for name in sorted(names):
        domain, category, _ = safe_relative(name).split("/")
        grouped[domain][category].append(name)
    splits = {}
    for domain in DOMAINS:
        training, validation, testing = [], [], []
        for label, category in enumerate(CLASSES):
            values = grouped[domain][category]
            if len(values) < 2:
                raise ValueError(f"Need at least two images in {domain}/{category}")
            ranked = sorted(
                values,
                key=lambda name: hashlib.sha256(
                    f"{SPLIT_POLICY['split_seed']}:{domain}:{category}:{name}".encode()
                ).hexdigest(),
            )
            holdout = max(
                1,
                len(values)
                * SPLIT_POLICY["validation_numerator"]
                // SPLIT_POLICY["validation_denominator"],
            )
            validation.extend([[name, label] for name in ranked[:holdout]])
            training.extend([[name, label] for name in ranked[holdout:]])
            testing.extend([[name, label] for name in values])
        splits[domain] = {
            "train": sorted(training),
            "crossval": sorted(validation),
            "test": sorted(testing),
        }
    return splits


def verify_original_image_collection(images):
    """Require the verified original image bytes even for manually imported folders."""
    reference = Path(__file__).resolve().parents[1] / "references/dataset_sources.json"
    expected = read_json(reference)["original_image_collection_sha256"]
    actual = canonical_hash({name: info["sha256"] for name, info in images.items()})
    if actual != expected:
        raise ValueError(
            "Image collection differs from the verified original-resolution Office-Home package"
        )
    return actual


def prepare_manifest(data_root, manifest_path, reference_dir=None, *, fixture=False):
    """Fully decode all images and bind exact paths, bytes, labels, and split policy."""
    if type(fixture) is not bool:
        raise ValueError("fixture must be a boolean")
    root = Path(data_root).resolve()
    discovered = discover_images(root)
    splits = build_splits(discovered)
    counts = {domain: len(splits[domain]["test"]) for domain in DOMAINS}
    if not fixture and counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Incomplete Office-Home package: expected {EXPECTED_COUNTS}, found {counts}"
        )
    images, dimensions = {}, Counter()
    for name, relative in discovered.items():
        path = checked_image_path(root, relative)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = image.size
            image.convert("RGB").load()
        dimensions[size] += 1
        images[name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
            "width": size[0],
            "height": size[1],
        }
    variant = "resized_256_square" if set(dimensions) == {(256, 256)} else "original_resolution"
    if not fixture and variant == "original_resolution":
        verify_original_image_collection(images)
    content = {
        "schema": 1,
        "kind": "officehome_fixture" if fixture else "officehome",
        "classes": list(CLASSES),
        "domains": list(DOMAINS),
        "variant": variant,
        "split_policy": SPLIT_POLICY,
        "splits": splits,
        "images": images,
    }
    manifest = {**content, "data_root": str(root), "dataset_id": canonical_hash(content)}
    atomic_json(manifest_path, manifest)
    return manifest


def load_manifest(path, reference_dir=None, verify_images=True):
    """Recompute the prescribed split and verify every image before training or evaluation."""
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("Unsupported Office-Home manifest schema")
    content = {
        key: value for key, value in manifest.items() if key not in ("dataset_id", "data_root")
    }
    if (
        canonical_hash(content) != manifest.get("dataset_id")
        or manifest.get("classes") != list(CLASSES)
        or manifest.get("domains") != list(DOMAINS)
        or manifest.get("split_policy") != SPLIT_POLICY
        or manifest.get("variant") not in ("original_resolution", "resized_256_square")
        or manifest.get("kind") not in ("officehome", "officehome_fixture")
    ):
        raise ValueError("Office-Home manifest identity, labels, or policy changed")
    if manifest["splits"] != build_splits(manifest["images"]):
        raise ValueError("Office-Home split differs from the prespecified partition")
    if manifest["kind"] == "officehome" and manifest["variant"] == "original_resolution":
        verify_original_image_collection(manifest["images"])
    if (
        manifest["kind"] == "officehome"
        and {domain: len(manifest["splits"][domain]["test"]) for domain in DOMAINS}
        != EXPECTED_COUNTS
    ):
        raise ValueError("Office-Home image counts differ from the complete benchmark")
    for name, info in manifest["images"].items():
        if canonical_name(info["path"]) != name:
            raise ValueError("Physical path and canonical Office-Home label disagree")
        if verify_images:
            image_path = checked_image_path(manifest["data_root"], info["path"])
            if (
                image_path.stat().st_size != info["bytes"]
                or file_hash(image_path) != info["sha256"]
            ):
                raise ValueError(f"Office-Home image content changed or is missing: {name}")
    return manifest


def validate_manifest_config(manifest, config):
    """Prevent accidental mixing of original-resolution and resized image packages."""
    if manifest["kind"] == "officehome" and manifest["variant"] != config.data_variant:
        raise ValueError(
            f"Image variant is {manifest['variant']}; config requires {config.data_variant}. Use an explicit matching config."
        )


def validate_archive_member(name, mode=0):
    """Reject traversal, special files, symbolic links, and ambiguous archive paths."""
    path = PurePosixPath(name.rstrip("/"))
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or not path.parts
        or path.as_posix() != name.rstrip("/")
        or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)))
    ):
        raise ValueError(f"Unsafe Office-Home archive member: {name}")
    return path


def extract_archive(archive, destination):
    """Stage ZIP or TGZ inputs and publish exactly one complete image root atomically."""
    archives = (
        [Path(archive)] if isinstance(archive, (str, Path)) else [Path(path) for path in archive]
    )
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen, total = set(), 0
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        for path in archives:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as handle:
                    for member in handle.infolist():
                        relative = validate_archive_member(
                            member.filename, member.external_attr >> 16
                        )
                        total += member.file_size
                        if total > MAX_ARCHIVE_BYTES or (
                            not member.is_dir() and relative.as_posix() in seen
                        ):
                            raise ValueError("Oversized archive or duplicate file member")
                        if member.flag_bits & 1:
                            raise ValueError("Encrypted dataset archives are unsupported")
                        if not member.is_dir():
                            seen.add(relative.as_posix())
                        target = temporary.joinpath(*relative.parts)
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with handle.open(member) as source, target.open("wb") as output:
                                shutil.copyfileobj(source, output)
            elif tarfile.is_tarfile(path):
                with tarfile.open(path, "r:*") as handle:
                    for member in handle:
                        relative = validate_archive_member(member.name)
                        if not (member.isfile() or member.isdir()):
                            raise ValueError("Links and special tar members are forbidden")
                        total += member.size
                        if total > MAX_ARCHIVE_BYTES or (
                            member.isfile() and relative.as_posix() in seen
                        ):
                            raise ValueError("Oversized archive or duplicate file member")
                        target = temporary.joinpath(*relative.parts)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            seen.add(relative.as_posix())
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with handle.extractfile(member) as source, target.open("wb") as output:
                                shutil.copyfileobj(source, output)
            else:
                raise ValueError(f"Not a supported ZIP/TGZ archive: {path}")
        candidates = [temporary] + [
            item for pattern in ("*", "*/*") for item in temporary.glob(pattern) if item.is_dir()
        ]
        required = {folder_key(domain) for domain in DOMAINS}
        roots = [
            item
            for item in candidates
            if required <= {folder_key(child.name) for child in item.iterdir() if child.is_dir()}
        ]
        if len(roots) != 1:
            raise ValueError("Cannot identify exactly one complete Office-Home image root")
        discover_images(roots[0])
        shutil.move(str(roots[0]), str(destination))
    return {
        "archives": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in archives]
    }


def download_officehome(directory, source_reference):
    """Fetch pinned original-resolution domain archives; never fall back to resized pixels."""
    import requests

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = read_json(source_reference)["original_domain_archives"]
    archives = []
    for domain in DOMAINS:
        specification = sources[domain]
        path = directory / f"{domain}.tgz"
        if path.exists():
            if file_hash(path) != specification["sha256"]:
                raise ValueError(f"Existing cached Office-Home archive hash differs: {path}")
            archives.append(path)
            print(f"Verified cached Office-Home archive: {domain}", flush=True)
            continue
        temporary = path.with_suffix(".tgz.partial")
        print(f"Downloading original-resolution Office-Home archive: {domain}", flush=True)
        try:
            with requests.get(specification["url"], stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            if file_hash(temporary) != specification["sha256"]:
                raise ValueError(f"Downloaded Office-Home hash differs: {domain}")
            os.replace(temporary, path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            atomic_json(
                directory / "download_failure.json",
                {
                    "domain": domain,
                    "url": specification["url"],
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise RuntimeError(
                "Office-Home download failed; import an existing original-resolution ZIP with prepare --archive or use prepare --data-root"
            ) from error
        archives.append(path)
    return archives, {
        "source": "Transfer-Learning-Library original-resolution mirror",
        "urls": [sources[domain]["url"] for domain in DOMAINS],
    }


class OfficeHomeDataset(Dataset):
    """Open only the explicitly requested source or sealed target partition."""

    def __init__(self, manifest, domains, split, size, training=False):
        """Resolve canonical identities while retaining the manifest's physical path mapping."""
        if (
            not domains
            or len(set(domains)) != len(domains)
            or any(domain not in DOMAINS for domain in domains)
            or split not in ("train", "crossval", "test")
        ):
            raise ValueError("Invalid Office-Home domains or partition")
        self.manifest = manifest
        self.records = []
        self.domain_indices = {}
        for domain in domains:
            self.domain_indices[domain] = []
            for name, label in manifest["splits"][domain][split]:
                self.domain_indices[domain].append(len(self.records))
                self.records.append((name, label))
        self.transform = image_transform(size, training)
        self.training = training

    def __len__(self):
        """Return the exact selected partition size."""
        return len(self.records)

    def __getitem__(self, index):
        """Apply independently seeded augmentation without advancing training RNGs."""
        seed = None
        if isinstance(index, tuple):
            index, seed = index
        if self.training and seed is None:
            raise ValueError("Training requires a stateless augmentation seed")
        name, label = self.records[index]
        path = checked_image_path(self.manifest["data_root"], self.manifest["images"][name]["path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            if seed is None:
                return self.transform(image), label
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.random.default_generator.manual_seed(seed)
                    return self.transform(image), label
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


def training_loader(manifest, config, start_step=0, end_step=None):
    """Balance all three source domains using the unchanged stateless step sampler."""
    dataset = OfficeHomeDataset(
        manifest,
        [domain for domain in DOMAINS if domain != config.target],
        "train",
        config.image_size,
        training=True,
    )
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
        generator=torch.Generator().manual_seed(keyed_seed(config.seed, "officehome_loader")),
    )


def evaluation_loader(manifest, domain, split, config):
    """Use deterministic unaugmented evaluation with an isolated loader RNG."""
    return DataLoader(
        OfficeHomeDataset(manifest, [domain], split, config.image_size),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.device == "cuda",
        generator=torch.Generator().manual_seed(0),
    )


def make_fixture(directory):
    """Create a clearly marked 65-class artificial package for protocol tests only."""
    directory = Path(directory)
    for domain_index, domain in enumerate(DOMAINS):
        for label, category in enumerate(CLASSES):
            folder = directory / domain / category
            folder.mkdir(parents=True, exist_ok=True)
            for number in range(2):
                image = Image.new(
                    "RGB",
                    (64, 64),
                    (
                        (label * 31 + number * 17) % 256,
                        (domain_index * 61 + label * 7) % 256,
                        (number * 91 + label * 13) % 256,
                    ),
                )
                image.save(folder / f"{number:06d}.jpg")
    path = directory.parent / "fixture_manifest.json"
    prepare_manifest(directory, path, fixture=True)
    return path
