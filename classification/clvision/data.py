"""Verified datasets, fixed source-only splits, and deterministic per-example transforms."""

import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json

from .config import CORRUPTIONS, DATASETS

TINY_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
TINY_MIRROR = "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1"
CORRUPTION_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
CORRUPTION_MD5 = "11f0ed0f1191edbf9fa23466ae6021d3"
NORMALIZATION = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tiny_imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dominoes": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def key_seed(*parts):
    """Create a stable CPU-local seed without changing worker or CUDA RNG state."""
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"
    ) % (2**63)


def relative_file(root, name):
    """Reject traversal and symlinks for every file described by a manifest."""
    root = Path(root).resolve()
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or not pure.parts:
        raise ValueError("Unsafe relative dataset path")
    path = root.joinpath(*pure.parts)
    for ancestor in (path, *path.parents):
        if ancestor == root.parent:
            break
        if ancestor.is_symlink():
            raise ValueError("Dataset symlinks are not supported")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Dataset path escapes root")
    return path


def digest_md5(path):
    """Stream the release checksum for an archive without loading it into memory."""
    value = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url, destination, md5):
    """Resume network interruptions only with checked ranges, then verify the complete archive."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_md5(destination) != md5:
            raise ValueError(f"Archive checksum mismatch: {destination}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists() and digest_md5(partial) == md5:
        os.replace(partial, destination)
        return destination
    for attempt in range(3):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "cog-vision-model/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = start > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {start}-"
                ):
                    raise ValueError("Download server returned an incorrect resume range")
                mode = "ab" if append else "wb"
                total = announced = start if append else 0
                with partial.open(mode) as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        total += len(block)
                        if total - announced >= 128 * 1024 * 1024:
                            print(
                                f"Downloading {destination.name}: {total / 2**20:.0f} MiB",
                                flush=True,
                            )
                            announced = total
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    if digest_md5(partial) != md5:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Downloaded archive checksum mismatch: {destination.name}")
    os.replace(partial, destination)
    return destination


def extract_archive(archive, destination, max_bytes=5 * 2**30):
    """Extract regular files only into a new directory, rejecting traversal and links."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Extraction never replaces an existing dataset directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        total = 0
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                seen = set()
                for info in source.infolist():
                    path = relative_file(temporary, info.filename.rstrip("/"))
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or info.filename in seen:
                        raise ValueError("Links, special files, or duplicate archive names")
                    seen.add(info.filename)
                    total += info.file_size
                    if total > max_bytes:
                        raise ValueError("Archive exceeds declared extraction budget")
                    if info.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with source.open(info) as stream, path.open("wb") as output:
                            shutil.copyfileobj(stream, output)
        else:
            with tarfile.open(archive) as source:
                seen = set()
                for info in source:
                    path = relative_file(temporary, info.name.rstrip("/"))
                    if not (info.isfile() or info.isdir()) or info.name in seen:
                        raise ValueError("Links, special files, or duplicate archive names")
                    seen.add(info.name)
                    total += info.size
                    if total > max_bytes:
                        raise ValueError("Archive exceeds declared extraction budget")
                    if info.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(info) as stream, path.open("wb") as output:
                            shutil.copyfileobj(stream, output)
        os.replace(temporary, destination)
        temporary.mkdir()


def download_mirrors(urls, destination, md5):
    """Fallback only for unavailable endpoints, always requiring identical archive bytes."""
    errors = []
    if Path(destination).exists():
        if digest_md5(destination) != md5:
            raise ValueError("Existing archive checksum mismatch")
        return Path(destination)
    for url in urls:
        try:
            value = download(url, destination, md5)
            atomic_json(
                str(destination) + ".source.json",
                {"url": url, "md5": md5, "sha256": file_hash(value)},
            )
            return value
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            errors.append(f"{url}: {error}")
            print(
                f"Download endpoint unavailable: {url}; checking the next identical-byte mirror",
                flush=True,
            )
    raise RuntimeError("No verified archive endpoint available: " + "; ".join(errors))


def stratified_split(labels, classes, per_class, seed=0):
    """Fix component identities before any training seed or method is considered."""
    train, validation = [], []
    for label in range(classes):
        indices = np.flatnonzero(labels == label).tolist()
        indices.sort(key=lambda index: hashlib.sha256(f"{seed}|{label}|{index}".encode()).digest())
        if len(indices) <= per_class:
            raise ValueError("Not enough class examples for the prescribed validation split")
        validation.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return sorted(train), sorted(validation)


def save_array(root, name, value):
    """Write a portable array once and bind its complete bytes to the manifest."""
    path = relative_file(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
        if path.exists() and file_hash(path) != file_hash(temporary):
            raise ValueError("Existing converted dataset bytes differ; use a fresh data root")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "path": name,
        "sha256": file_hash(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def seal_manifest(root, payload):
    """Keep semantic identity stable when the whole dataset root is relocated."""
    payload = {"schema": 1, **payload}
    value = {**payload, "dataset_id": canonical_hash(payload), "root": str(Path(root).resolve())}
    path = Path(root) / "manifests" / f"{payload['dataset']}.json"
    if path.exists() and read_json(path)["dataset_id"] != value["dataset_id"]:
        raise ValueError("Preparation would change an existing dataset identity")
    atomic_json(path, value)
    return value


def prepare_cifar(root, name="cifar100", download_data=False):
    """Use torchvision's official checked archives and persist fine-label image arrays."""
    root = Path(root)
    if name not in ("cifar100", "cifar10"):
        raise ValueError("Unknown CIFAR release")
    factory = CIFAR100 if name == "cifar100" else CIFAR10
    raw_root = root / "raw" / name
    if not (raw_root / factory.base_folder).is_dir():
        checked = raw_root / "checked"
        if not checked.exists():
            archive = raw_root / factory.filename
            if download_data:
                mirror = f"https://dataset.bj.bcebos.com/cifar/{factory.filename}"
                download_mirrors((mirror, factory.url), archive, factory.tgz_md5)
            if not archive.exists() or digest_md5(archive) != factory.tgz_md5:
                raise ValueError("Verified CIFAR archive is required; use prepare --download")
            extract_archive(archive, checked)
        raw_root = checked
    datasets = [factory(raw_root, train=flag, download=False) for flag in (True, False)]
    classes = 100 if name == "cifar100" else 10
    arrays = {}
    for split, dataset, expected in zip(("train", "test"), datasets, (50000, 10000)):
        labels = np.asarray(dataset.targets, dtype=np.int64)
        images = dataset.data
        if images.shape != (expected, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Official CIFAR shape or dtype mismatch")
        if np.bincount(labels, minlength=classes).tolist() != [expected // classes] * classes:
            raise ValueError("Official CIFAR class counts differ")
        arrays[f"{split}_images"] = save_array(root, f"arrays/{name}_{split}_images.npy", images)
        arrays[f"{split}_labels"] = save_array(root, f"arrays/{name}_{split}_labels.npy", labels)
    train, validation = stratified_split(
        np.asarray(datasets[0].targets), classes, 50 if classes == 100 else 500
    )
    return seal_manifest(
        root,
        {
            "dataset": name,
            "kind": "official",
            "classes": datasets[0].classes,
            "arrays": arrays,
            "splits": {"train": train, "validation": validation, "test": list(range(10000))},
            "split_policy": "sha256(class,index,split_seed0), fixed stratified validation; original official test",
            "shape": [32, 32],
        },
    )


def prepare_tiny(root, download_data=False):
    """Verify all200 classes and decode/hash every labeled native-resolution image."""
    root = Path(root)
    extracted = root / "raw" / "tiny"
    if not extracted.exists():
        archive = root / "archives" / "tiny-imagenet-200.zip"
        if download_data:
            download_mirrors((TINY_URL, TINY_MIRROR), archive, TINY_MD5)
        if not archive.exists() or digest_md5(archive) != TINY_MD5:
            raise ValueError("A verified official Tiny ImageNet archive is required")
        extract_archive(archive, extracted)
    folder = extracted / "tiny-imagenet-200"
    classes = (folder / "wnids.txt").read_text().splitlines()
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("Tiny ImageNet must contain200 unique classes")
    classes = sorted(classes)
    class_to_index = {name: index for index, name in enumerate(classes)}
    records, labels = [], []
    for label, name in enumerate(classes):
        paths = sorted((folder / "train" / name / "images").glob("*.JPEG"))
        if len(paths) != 500:
            raise ValueError(f"Tiny training count mismatch for{name}")
        for path in paths:
            with Image.open(path) as image:
                if image.size != (64, 64):
                    raise ValueError("Tiny images must retain native64x64 resolution")
                image.convert("RGB").load()
            records.append([path.relative_to(root).as_posix(), label, file_hash(path)])
            labels.append(label)
    train, validation = stratified_split(np.asarray(labels), 200, 50)
    test = []
    names = set()
    for line in (folder / "val" / "val_annotations.txt").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or fields[0] in names or fields[1] not in class_to_index:
            raise ValueError("Invalid Tiny final validation annotations")
        names.add(fields[0])
        path = relative_file(folder / "val" / "images", fields[0])
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError("Tiny final image resolution changed")
            image.convert("RGB").load()
        test.append([path.relative_to(root).as_posix(), class_to_index[fields[1]], file_hash(path)])
    if (
        len(test) != 10000
        or np.bincount([row[1] for row in test], minlength=200).tolist() != [50] * 200
    ):
        raise ValueError("Tiny final validation count differs")
    print("Verified110,000 labeled Tiny ImageNet images", flush=True)
    return seal_manifest(
        root,
        {
            "dataset": "tiny_imagenet",
            "kind": "official",
            "classes": classes,
            "records": records,
            "test_records": sorted(test),
            "splits": {"train": train, "validation": validation, "test": list(range(10000))},
            "shape": [64, 64],
            "archive_md5": TINY_MD5,
            "split_policy": "50/class internal validation from train; official validation sealed as final test",
        },
    )


def prepare_dominoes(root, download_data=False):
    """Separate digit/vehicle component identities before building the95%-aligned variant."""
    root = Path(root)
    cifar_path = root / "manifests" / "cifar10.json"
    if not cifar_path.exists():
        prepare_cifar(root, "cifar10", download_data)
    cifar = load_manifest(cifar_path, root_override=root)
    arrays = dict(cifar["arrays"])
    digit_splits = {}
    for split, flag in (("train", True), ("test", False)):
        dataset = MNIST(root / "raw" / "mnist", train=flag, download=download_data)
        labels = dataset.targets.numpy().astype(np.int64)
        images = dataset.data.numpy()
        selected = np.flatnonzero(labels < 2)
        images, labels = images[selected], labels[selected]
        arrays[f"digit_{split}_images"] = save_array(
            root, f"arrays/digit_{split}_images.npy", images
        )
        arrays[f"digit_{split}_labels"] = save_array(
            root, f"arrays/digit_{split}_labels.npy", labels
        )
        if split == "train":
            digit_train, digit_validation = [], []
            for label in (0, 1):
                indices = np.flatnonzero(labels == label).tolist()
                indices.sort(key=lambda index: key_seed("digit-split", label, index))
                boundary = len(indices) // 10
                digit_validation.extend(indices[:boundary])
                digit_train.extend(indices[boundary:])
            digit_splits.update(train=sorted(digit_train), validation=sorted(digit_validation))
        else:
            digit_splits["test"] = list(range(len(labels)))
    vehicle_train_labels = read_array(root, arrays["train_labels"])
    vehicle_test_labels = read_array(root, arrays["test_labels"])
    splits = {}
    for split, indices in cifar["splits"].items():
        labels = vehicle_test_labels if split == "test" else vehicle_train_labels
        splits[split] = [index for index in indices if labels[index] in (1, 9)]
    if [len(splits[key]) for key in ("train", "validation", "test")] != [9000, 1000, 2000]:
        raise ValueError("Dominoes vehicle component counts differ")
    return seal_manifest(
        root,
        {
            "dataset": "dominoes",
            "kind": "official_component_variant",
            "classes": ["automobile", "truck"],
            "arrays": arrays,
            "splits": splits,
            "digit_splits": digit_splits,
            "shape": [64, 32],
            "split_policy": "Components split before pairing; train and validation95% agreement; paired final counterfactuals",
            "agreement": 0.95,
            "test_agreements": [0.95, 0.5, 0.05],
        },
    )


def prepare_corruptions(root, download_data=False):
    """Pin released corruptions and verify every severity label against the clean test order."""
    root = Path(root)
    clean = load_manifest(root / "manifests" / "cifar100.json", root_override=root)
    extracted = root / "raw" / "corruptions"
    if not extracted.exists():
        archive = root / "archives" / "CIFAR-100-C.tar"
        if download_data:
            download(CORRUPTION_URL, archive, CORRUPTION_MD5)
        if not archive.exists() or digest_md5(archive) != CORRUPTION_MD5:
            raise ValueError("Official CIFAR-100-C archive with matching MD5 is required")
        extract_archive(archive, extracted)
    folder = extracted / "CIFAR-100-C"
    labels_path = folder / "labels.npy"
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    clean_labels = read_array(root, clean["arrays"]["test_labels"])
    expected = np.tile(clean_labels, 5) if labels.shape == (50000,) else clean_labels
    if labels.shape not in ((10000,), (50000,)) or not np.array_equal(labels, expected):
        raise ValueError("Corruption labels do not match the clean test ordering")
    files = {}
    for name in CORRUPTIONS:
        path = folder / f"{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (50000, 32, 32, 3) or values.dtype != np.uint8:
            raise ValueError(f"Corruption shape/dtype mismatch: {name}")
        files[name] = {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
    return seal_manifest(
        root,
        {
            "dataset": "cifar100_c",
            "kind": "official",
            "clean_dataset_id": clean["dataset_id"],
            "archive_md5": CORRUPTION_MD5,
            "corruptions": files,
            "severities": [1, 2, 3, 4, 5],
            "labels": {
                "path": labels_path.relative_to(root).as_posix(),
                "sha256": file_hash(labels_path),
            },
            "shape": [32, 32],
        },
    )


def read_array(root, descriptor):
    """Memory-map a verified non-pickle array without allocating a full dataset copy."""
    return np.load(relative_file(root, descriptor["path"]), mmap_mode="r", allow_pickle=False)


def load_manifest(path, verify_bytes=True, root_override=None):
    """Verify manifest identity, paths, and exact image/array bytes before use."""
    value = read_json(path)
    payload = {key: item for key, item in value.items() if key not in ("dataset_id", "root")}
    if value.get("schema") != 1 or canonical_hash(payload) != value.get("dataset_id"):
        raise ValueError("Dataset manifest identity mismatch")
    root = Path(root_override or value["root"]).resolve()
    value["root"] = str(root)
    files = list(value.get("arrays", {}).values())
    files += list(value.get("corruptions", {}).values())
    if "labels" in value:
        files.append(value["labels"])
    for item in files:
        file = relative_file(root, item["path"])
        if not file.is_file() or (verify_bytes and file_hash(file) != item["sha256"]):
            raise ValueError(f"Dataset array changed: {file}")
        if "shape" in item:
            array = read_array(root, item)
            if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
                raise ValueError("Dataset array schema changed")
    for row in value.get("records", []) + value.get("test_records", []):
        file = relative_file(root, row[0])
        if not file.is_file() or (verify_bytes and file_hash(file) != row[2]):
            raise ValueError(f"Dataset image changed: {file}")
    if "splits" in value:
        train, validation = value["splits"]["train"], value["splits"]["validation"]
        if (
            len(set(train)) != len(train)
            or len(set(validation)) != len(validation)
            or set(train) & set(validation)
        ):
            raise ValueError("Overlapping or duplicate training/validation identities")
        if value["dataset"] in DATASETS and len(value["classes"]) != DATASETS[value["dataset"]][0]:
            raise ValueError("Class order differs from the declared benchmark")
        if any(
            type(index) is not int or index < 0
            for indices in value["splits"].values()
            for index in indices
        ):
            raise ValueError("Invalid partition index")
        if any(len(set(indices)) != len(indices) for indices in value["splits"].values()):
            raise ValueError("Duplicate partition identities")
        if "records" in value:
            labels = np.asarray([row[1] for row in value["records"]])
            test_n = len(value["test_records"])
        else:
            labels = read_array(root, value["arrays"]["train_labels"])
            test_n = len(read_array(root, value["arrays"]["test_labels"]))
        if (
            max(train + validation, default=-1) >= len(labels)
            or max(value["splits"]["test"], default=-1) >= test_n
        ):
            raise ValueError("Partition index exceeds source array length")
        if value["dataset"] == "dominoes":
            digits = value["digit_splits"]
            if set(digits["train"]) & set(digits["validation"]):
                raise ValueError("Digit component identities leak between train and validation")
            for split, indices in digits.items():
                source = "test" if split == "test" else "train"
                array = read_array(root, value["arrays"][f"digit_{source}_labels"])
                if (
                    len(set(indices)) != len(indices)
                    or not indices
                    or min(indices) < 0
                    or max(indices) >= len(array)
                ):
                    raise ValueError("Invalid digit component identities")
            if value["kind"] != "fixture" and [
                len(value["splits"][split]) for split in ("train", "validation", "test")
            ] != [9000, 1000, 2000]:
                raise ValueError("Production Dominoes partition counts changed")
        elif value["kind"] != "fixture":
            classes = len(value["classes"])
            expected = (
                50000
                if value["dataset"] == "cifar100"
                else 100000
                if value["dataset"] == "tiny_imagenet"
                else 50000
            )
            per_class = 50 if classes in (100, 200) else 500
            expected_train, expected_validation = stratified_split(labels, classes, per_class)
            if (
                len(labels) != expected
                or train != expected_train
                or validation != expected_validation
                or value["splits"]["test"] != list(range(10000))
            ):
                raise ValueError("Production partition differs from the prescribed split")
    return value


def transform_image(image, dataset, training=False, seed=0, tiny_imagenet_size=64):
    """Resize Tiny before keyed augmentation; preserve native CIFAR and Dominoes processing."""
    if type(tiny_imagenet_size) is not int or tiny_imagenet_size not in (32, 64):
        raise ValueError("tiny_imagenet_size must be 32 or 64")
    value = np.array(image, dtype=np.uint8, copy=True)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Expected HWC RGB image")
    if dataset == "tiny_imagenet" and tiny_imagenet_size == 32:
        value = np.asarray(Image.fromarray(value).resize((32, 32), Image.Resampling.BILINEAR))
    rng = np.random.default_rng(seed)
    if training and dataset != "dominoes":
        padding = 4 if dataset == "cifar100" else tiny_imagenet_size // 8
        height, width = value.shape[:2]
        padded = np.pad(value, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
        y, x = rng.integers(0, 2 * padding + 1, size=2)
        value = padded[y : y + height, x : x + width]
        if rng.random() < 0.5:
            value = value[:, ::-1]
    tensor = torch.from_numpy(value.copy()).permute(2, 0, 1).float().div_(255)
    mean, deviation = NORMALIZATION[dataset]
    return (tensor - tensor.new_tensor(mean)[:, None, None]) / tensor.new_tensor(deviation)[
        :, None, None
    ]


class ImageDataset(Dataset):
    """Expose explicitly requested partitions; training never constructs a final-test dataset."""

    def __init__(self, manifest, split, seed=0, epoch=0, diagnostic="both", tiny_imagenet_size=64):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown partition")
        self.manifest, self.split, self.seed, self.epoch = manifest, split, seed, epoch
        self.root, self.dataset = Path(manifest["root"]), manifest["dataset"]
        self.indices = manifest["splits"][split]
        self.arrays = {}
        self.diagnostic = diagnostic
        if type(tiny_imagenet_size) is not int or tiny_imagenet_size not in (32, 64):
            raise ValueError("tiny_imagenet_size must be 32 or 64")
        self.tiny_imagenet_size = tiny_imagenet_size
        if diagnostic not in ("both", "vehicle_only", "digit_only") or (
            diagnostic != "both" and split != "test"
        ):
            raise ValueError("Panel deletion is a final-test diagnostic only")
        if "records" not in manifest:
            self.arrays = {
                key: read_array(self.root, item) for key, item in manifest["arrays"].items()
            }
        if self.dataset == "dominoes":
            source = "test" if split == "test" else "train"
            digit_labels = self.arrays[f"digit_{source}_labels"]
            digit_indices = manifest["digit_splits"][split]
            self.digit_pools = {
                label: [index for index in digit_indices if digit_labels[index] == label]
                for label in (0, 1)
            }
            if not all(self.digit_pools.values()):
                raise ValueError("Empty digit component pool")
            self.digit_assignments = {}
            if split != "test":
                labels = self.arrays["train_labels"]
                for label in (0, 1):
                    indices = [index for index in self.indices if int(labels[index] == 9) == label]
                    indices.sort(key=lambda index: key_seed("alignment", split, label, index))
                    boundary = round(manifest["agreement"] * len(indices))
                    self.digit_assignments.update(
                        {
                            index: label if slot < boundary else 1 - label
                            for slot, index in enumerate(indices)
                        }
                    )

    def __len__(self):
        """Counterfactual final testing pairs each vehicle with both digit classes."""
        return len(self.indices) * (2 if self.dataset == "dominoes" and self.split == "test" else 1)

    def sample_id(self, index):
        """Provide stable identifiers for prediction alignment and paired reports."""
        if self.dataset == "dominoes" and self.split == "test":
            return f"vehicle:{self.indices[index // 2]}|digit:{index % 2}"
        index = self.indices[index]
        if "records" in self.manifest:
            return (
                self.manifest["test_records"] if self.split == "test" else self.manifest["records"]
            )[index][0]
        return f"{self.dataset}:{'test' if self.split == 'test' else 'train'}:{index}"

    def sample_target(self, item):
        """Read canonical labels without decoding images or applying augmentations."""
        index = self.indices[
            item // 2 if self.dataset == "dominoes" and self.split == "test" else item
        ]
        if "records" in self.manifest:
            records = (
                self.manifest["test_records"] if self.split == "test" else self.manifest["records"]
            )
            return int(records[index][1])
        source = "test" if self.split == "test" else "train"
        target = int(self.arrays[f"{source}_labels"][index])
        return int(target == 9) if self.dataset == "dominoes" else target

    def __getitem__(self, item):
        """Construct one normalized RGB tensor with deterministic component pairing."""
        index = self.indices[
            item // 2 if self.dataset == "dominoes" and self.split == "test" else item
        ]
        source = "test" if self.split == "test" else "train"
        if "records" in self.manifest:
            row = (
                self.manifest["test_records"] if self.split == "test" else self.manifest["records"]
            )[index]
            with Image.open(relative_file(self.root, row[0])) as handle:
                image = np.array(handle.convert("RGB"))
            target = row[1]
        elif self.dataset == "dominoes":
            target = int(self.arrays[f"{source}_labels"][index] == 9)
            digit = item % 2 if self.split == "test" else self.digit_assignments[index]
            pool = self.digit_pools[digit]
            pairing_epoch = self.epoch if self.split == "train" else 0
            identity = key_seed("digit-pair", self.split, pairing_epoch, index, digit) % len(pool)
            digit_image = self.arrays[f"digit_{source}_images"][pool[identity]]
            top = np.repeat(np.pad(digit_image, 2)[:, :, None], 3, axis=2)
            bottom = self.arrays[f"{source}_images"][index]
            if self.diagnostic == "vehicle_only":
                top = np.zeros_like(top)
            if self.diagnostic == "digit_only":
                bottom = np.zeros_like(bottom)
            image = np.concatenate((top, bottom), axis=0)
        else:
            image, target = (
                self.arrays[f"{source}_images"][index],
                int(self.arrays[f"{source}_labels"][index]),
            )
        augment_seed = key_seed("augment", self.seed, self.epoch, self.sample_id(item))
        return transform_image(
            image, self.dataset, self.split == "train", augment_seed, self.tiny_imagenet_size
        ), target


class EpochBatches(Sampler):
    """Generate one matched no-replacement epoch without consuming model RNG state."""

    def __init__(self, length, batch_size, seed, epoch, limit=0):
        self.length, self.batch_size, self.seed, self.epoch, self.limit = (
            length,
            batch_size,
            seed,
            epoch,
            limit,
        )

    def __iter__(self):
        """Use an independent CPU generator to fix the full sample order."""
        generator = torch.Generator().manual_seed(key_seed("sample-order", self.seed, self.epoch))
        order = torch.randperm(self.length, generator=generator).tolist()
        for slot, start in enumerate(range(0, self.length, self.batch_size)):
            if self.limit and slot >= self.limit:
                break
            yield order[start : start + self.batch_size]

    def __len__(self):
        """Report exactly the accepted number of batches in the epoch."""
        length = (self.length + self.batch_size - 1) // self.batch_size
        return min(length, self.limit) if self.limit else length


def make_loader(manifest, config, split, epoch=0, diagnostic="both"):
    """Keep worker RNGs isolated and require an explicit partition at every call."""
    dataset = ImageDataset(
        manifest, split, config.seed, epoch, diagnostic, config.tiny_imagenet_size
    )
    generator = torch.Generator().manual_seed(key_seed("workers", config.seed, epoch, split))
    options = {
        "num_workers": config.workers,
        "pin_memory": config.device == "cuda",
        "generator": generator,
    }
    if split == "train":
        return DataLoader(
            dataset,
            batch_sampler=EpochBatches(
                len(dataset), config.batch_size, config.seed, epoch, config.debug_max_batches
            ),
            **options,
        )
    return DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False, **options)


class CorruptionDataset(Dataset):
    """Read only one released corruption/severity with the clean test label ordering."""

    def __init__(self, manifest, clean, corruption, severity):
        if corruption not in CORRUPTIONS or severity not in (1, 2, 3, 4, 5):
            raise ValueError("Unregistered corruption or severity")
        if manifest["clean_dataset_id"] != clean["dataset_id"]:
            raise ValueError("Corruption and clean datasets do not match")
        root = Path(manifest["root"])
        self.images = np.load(
            relative_file(root, manifest["corruptions"][corruption]["path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.labels = read_array(clean["root"], clean["arrays"]["test_labels"])
        self.offset = (severity - 1) * len(self.labels)

    def __len__(self):
        """Keep each severity's denominator equal to the full clean test set."""
        return len(self.labels)

    def __getitem__(self, index):
        """Apply the unchanged clean-model normalization without adaptation."""
        return transform_image(self.images[self.offset + index], "cifar100"), int(
            self.labels[index]
        )


def make_fixture(root, dataset="cifar100", samples=12):
    """Create small labeled arrays for explicit debug verification, never benchmark evidence."""
    root = Path(root)
    classes, height, width = DATASETS[dataset]
    rng = np.random.default_rng(375)
    arrays = {}
    for split in ("train", "test"):
        if dataset == "dominoes":
            labels = np.where(np.arange(samples) % 2, 9, 1).astype(np.int64)
            images = rng.integers(0, 256, size=(samples, 32, 32, 3), dtype=np.uint8)
            arrays[f"{split}_images"] = save_array(
                root, f"fixture/{dataset}_{split}_images.npy", images
            )
            arrays[f"{split}_labels"] = save_array(
                root, f"fixture/{dataset}_{split}_labels.npy", labels
            )
            arrays[f"digit_{split}_images"] = save_array(
                root,
                f"fixture/digit_{split}_images.npy",
                rng.integers(0, 256, size=(samples, 28, 28), dtype=np.uint8),
            )
            arrays[f"digit_{split}_labels"] = save_array(
                root, f"fixture/digit_{split}_labels.npy", (np.arange(samples) % 2).astype(np.int64)
            )
            continue
        labels = (np.arange(samples, dtype=np.int64) * 17) % classes
        labels[-1] = classes - 1
        images = rng.integers(0, 256, size=(samples, height, width, 3), dtype=np.uint8)
        arrays[f"{split}_images"] = save_array(
            root, f"fixture/{dataset}_{split}_images.npy", images
        )
        arrays[f"{split}_labels"] = save_array(
            root, f"fixture/{dataset}_{split}_labels.npy", labels
        )
    payload = {
        "dataset": dataset,
        "kind": "fixture",
        "classes": [f"class{index}" for index in range(classes)],
        "arrays": arrays,
        "splits": {
            "train": list(range(samples - 4)),
            "validation": list(range(samples - 4, samples)),
            "test": list(range(samples)),
        },
        "shape": [height, width],
        "split_policy": "synthetic verification only",
    }
    if dataset == "dominoes":
        payload.update(
            digit_splits=dict(payload["splits"]), agreement=0.95, test_agreements=[0.95, 0.5, 0.05]
        )
    return seal_manifest(root, payload)
