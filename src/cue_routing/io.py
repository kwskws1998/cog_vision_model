"""Atomic artifacts, source fingerprints, deterministic RNGs, and run locks."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import tempfile
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout


def canonical_hash(value):
    """Hash canonical JSON for a plain experiment contract."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    """Stream a content hash without loading an entire checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Replace a JSON artifact only after its complete contents are flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path, value):
    """Atomically persist a trusted local checkpoint with tensor-safe contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    """Load a JSON artifact as ordinary data."""
    return json.loads(Path(path).read_text())


def state_hash(state):
    """Fingerprint exact model tensor values, names, shapes, and dtypes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hash(root):
    """Bind results to executable sources, split files, and the published recipe."""
    root = Path(root)
    paths = list((root / "src").rglob("*.py")) + list((root / "configs").glob("*.json"))
    paths += list((root / "references").rglob("*.txt")) + list((root / "tests").glob("*.py"))
    paths += list((root / "references").rglob("*.json"))
    paths += (
        list(root.glob("*.py"))
        + list(root.glob("*.yml"))
        + list(root.glob("requirements*.txt"))
        + [root / "pyproject.toml"]
    )
    return canonical_hash({str(path.relative_to(root)): file_hash(path) for path in sorted(paths)})


def seed_everything(seed, device="cpu"):
    """Seed training RNGs and disable unreported precision shortcuts."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def capture_rng(device="cpu"):
    """Serialize RNG states using objects accepted by weights_only loading."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            float(numpy_state[4]),
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
    }


def restore_rng(value):
    """Restore checkpoint RNGs after model and optimizer construction."""
    random.setstate(value["python"])
    numpy_state = value["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.array(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA checkpoint cannot be resumed as an exact CPU replay")
        torch.cuda.set_rng_state(value["cuda"])


def environment():
    """Record actual software and CUDA properties without claiming hardware equivalence."""
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "gdown", "filelock"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import sys

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "environment_manager": "conda"
        if (Path(sys.prefix) / "conda-meta" / "history").is_file()
        else "unmanaged",
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": properties.name,
            "total_gib": properties.total_memory / 2**30,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return result


@contextlib.contextmanager
def run_lock(directory):
    """Hold a kernel-backed writer lock that is automatically released on process death."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".running.json"
    lock = FileLock(str(directory / ".writer.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError("Another process is writing this run") from error
    try:
        atomic_json(path, {"pid": os.getpid(), "host": socket.gethostname()})
        yield
    finally:
        try:
            path.unlink(missing_ok=True)
        finally:
            lock.release()
