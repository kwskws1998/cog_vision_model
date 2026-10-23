"""Create or verify a project-local Conda environment from pinned requirements files."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

RUNTIME_PROBE = """
import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

import torch
import torchvision

prefix = Path(sys.prefix).resolve()
assert (prefix / "conda-meta" / "history").is_file(), "A Conda environment is required"
assert Path(torch.__file__).resolve().is_relative_to(prefix), "PyTorch is outside the environment"
assert Path(torchvision.__file__).resolve().is_relative_to(prefix), "torchvision is outside the environment"
bootstrap = runpy.run_path(sys.argv[3], run_name="environment_runtime_probe")
cuda = bootstrap["probe_cuda_runtime"](torch, sys.argv[2])
print(json.dumps({
    "prefix": str(prefix),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    **cuda,
    "dependencies": {name: importlib.metadata.version(name) for name in json.loads(sys.argv[1])},
}))
"""


def probe_cuda_runtime(torch, device):
    """Exercise real CUDA kernels and convolution gradients only for a requested CUDA setup."""
    report = {
        "cuda_available": False,
        "arch_list": [],
        "device_name": None,
        "device_capability": None,
        "kernel_checked": False,
        "kernel_test_passed": False,
    }
    if device != "cuda":
        report["kernel_skip_reason"] = "CPU environment requested"
        return report
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["arch_list"] = list(torch.cuda.get_arch_list())
    if not report["cuda_available"]:
        report["kernel_skip_reason"] = "CUDA GPU unavailable; GPU execution remains unverified"
        return report
    report["kernel_checked"] = True
    try:
        index = torch.cuda.current_device()
        report["device_name"] = torch.cuda.get_device_name(index)
        report["device_capability"] = list(torch.cuda.get_device_capability(index))
        target = f"cuda:{index}"
        inputs = torch.ones((2, 3, 8, 8), device=target, requires_grad=True)
        elementwise = (inputs * 2 + 1).sum()
        convolution = torch.nn.Conv2d(3, 4, 3, padding=1).to(target)
        loss = convolution(inputs).square().mean() + elementwise * 0.0001
        loss.backward()
        torch.cuda.synchronize(index)
        tensors = [loss, inputs.grad, *(parameter.grad for parameter in convolution.parameters())]
        if any(tensor is None or not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise RuntimeError(
                "CUDA convolution forward/backward produced missing or nonfinite values"
            )
        if elementwise.item() != inputs.numel() * 3:
            raise RuntimeError("CUDA elementwise kernel produced an incorrect result")
        report["kernel_test_passed"] = True
    except RuntimeError as error:
        report["kernel_test_error"] = str(error)
    return report


def resolve_conda(value=None):
    """Locate Conda without installing into or modifying the base environment."""
    candidate = value or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if candidate:
        candidate = shutil.which(str(candidate)) or candidate
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError(
            "Conda was not found; install Miniconda/Anaconda or pass --conda /absolute/path/to/conda"
        )
    return str(Path(candidate).resolve())


def dependency_pins(root, development=False):
    """Read exact runtime and optional development versions from their canonical files."""
    filenames = ["requirements.txt"]
    if development:
        filenames.append("requirements-dev.txt")
    pins = {}
    for filename in filenames:
        for line in (Path(root) / filename).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r ")):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line)
            if match is None:
                raise ValueError(f"Expected an exact package pin in {filename}: {line}")
            name, version = match.groups()
            if name in pins and pins[name] != version:
                raise ValueError(f"Conflicting dependency pin for {name}")
            pins[name] = version
    return pins


def wheel_specification(device, system, root=None):
    """Add official wheel build tags to the versions declared in requirements.txt."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if system not in ("Linux", "Darwin", "Windows"):
        raise ValueError(f"Unsupported operating system: {system}")
    if device == "cuda" and system != "Linux":
        raise ValueError("The CUDA benchmark environment is Linux; use --device cpu on this host")
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    pins = dependency_pins(root)
    build = "+cu128" if device == "cuda" else ("" if system == "Darwin" else "+cpu")
    index = f"https://download.pytorch.org/whl/{'cu128' if device == 'cuda' else 'cpu'}"
    return [f"{name}=={pins[name]}{build}" for name in ("torch", "torchvision")], index


def validate_prefix(prefix, root, base=None):
    """Reject non-environments and destructive reuse of project or base directories."""
    prefix, root = Path(prefix).resolve(), Path(root).resolve()
    if prefix == root or prefix in root.parents:
        raise ValueError("The Conda prefix must be a dedicated environment directory")
    if base is not None and prefix == Path(base).resolve():
        raise ValueError("Installing into the Conda base environment is forbidden")
    if prefix.exists() and not (prefix / "conda-meta" / "history").is_file():
        raise ValueError("Existing --prefix is not a Conda environment")
    return prefix


def validate_runtime(report, prefix, device, system, root, development=False):
    """Check exact wheel versions, dependency pins, Python, and the compiled CUDA runtime."""
    if Path(report["prefix"]).resolve() != Path(prefix).resolve():
        raise RuntimeError("The Python interpreter did not use the requested Conda environment")
    if tuple(report["python"].split(".")[:2]) != ("3", "11"):
        raise RuntimeError("The recipe requires Conda Python 3.11")
    expected = dependency_pins(root, development)
    packages, _ = wheel_specification(device, system, root)
    expected.update(dict(package.split("==", 1) for package in packages))
    for name, version in expected.items():
        actual = report["dependencies"].get(name)
        if actual != version:
            raise RuntimeError(f"Expected {name}=={version}; installed version is {actual}")
    for name in ("torch", "torchvision"):
        if report[name] != expected[name]:
            raise RuntimeError(f"Imported {name} version does not match its installed metadata")
    expected_cuda = "12.8" if device == "cuda" else None
    if report["cuda_runtime"] != expected_cuda:
        raise RuntimeError("Installed PyTorch build does not match the requested device")
    if device == "cuda" and report["cuda_available"]:
        if report.get("kernel_checked") is not True or report.get("kernel_test_passed") is not True:
            detail = report.get("kernel_test_error", "CUDA kernels were not successfully checked")
            raise RuntimeError(
                f"CUDA kernel verification failed on {report.get('device_name')} "
                f"(capability {report.get('device_capability')}; "
                f"wheel architectures {report.get('arch_list')}): {detail}. "
                "Install the pinned CUDA 12.8 requirements and verify NVIDIA driver compatibility."
            )
    elif report.get("kernel_checked") is not False or report.get("kernel_test_passed") is not False:
        raise RuntimeError("CUDA execution cannot be verified without a requested, available GPU")


def write_verification(root, prefix, device, conda, executable, environment, report):
    """Export the observed runtime and package lists without changing installed packages."""
    destination = root / "verification"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"conda_{device}_setup.json").write_text(json.dumps(report, indent=2) + "\n")
    explicit = subprocess.run(
        [conda, "list", "--prefix", str(prefix), "--explicit"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    (destination / f"conda_{device}_explicit.txt").write_text(explicit.stdout)
    frozen = subprocess.run(
        [str(executable), "-m", "pip", "freeze"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    (destination / f"{device}_environment_freeze.txt").write_text(frozen.stdout)


def install_environment(root, prefix, device, conda, development=False, verify_only=False):
    """Install requirements into dedicated Conda Python, or only verify an existing environment."""
    root = Path(root).resolve()
    system = platform.system()
    _, index = wheel_specification(device, system, root)
    requirements = root / f"requirements-{device}.txt"
    environment = {
        **os.environ,
        "PIP_CACHE_DIR": str(root / ".cache" / "pip"),
        "CONDA_PKGS_DIRS": str(root / ".cache" / "conda_pkgs"),
        "PYTHONNOUSERSITE": "1",
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    base = subprocess.run(
        [conda, "info", "--base"], check=True, env=environment, capture_output=True, text=True
    ).stdout.strip()
    prefix = validate_prefix(prefix, root, base)
    if not prefix.exists():
        if verify_only:
            raise ValueError("--verify-only requires an existing Conda environment")
        subprocess.run(
            [
                conda,
                "env",
                "create",
                "--yes",
                "--prefix",
                str(prefix),
                "--file",
                str(root / "environment.yml"),
            ],
            check=True,
            env=environment,
        )
    executable = prefix / ("python.exe" if system == "Windows" else "bin/python")
    subprocess.run(
        [
            str(executable),
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 11), 'The recipe requires Conda Python 3.11'",
        ],
        check=True,
        env=environment,
    )
    if not verify_only:
        command = [str(executable), "-m", "pip", "install", "-r", str(requirements)]
        if development:
            command.extend(["-r", str(root / "requirements-dev.txt")])
        subprocess.run(command, check=True, env=environment)
        subprocess.run(
            [str(executable), "-m", "pip", "install", "--no-deps", "-e", str(root)],
            check=True,
            env=environment,
        )
    subprocess.run([str(executable), "-m", "pip", "check"], check=True, env=environment)
    pins = dependency_pins(root, development)
    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                RUNTIME_PROBE,
                json.dumps(list(pins)),
                device,
                str(root / "bootstrap.py"),
            ],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        raise RuntimeError(f"Environment runtime probe failed: {detail.strip()}") from error
    report = json.loads(result.stdout)
    validate_runtime(report, prefix, device, system, root, development)
    report.update(
        {
            "environment_manager": "conda",
            "requested_device": device,
            "development_dependencies_checked": development,
            "verification_only": verify_only,
            "wheel_index": index,
            "requirements_file": requirements.name,
            "requirements_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.glob("requirements*.txt"))
            },
            "environment_yml_sha256": hashlib.sha256(
                (root / "environment.yml").read_bytes()
            ).hexdigest(),
            "bootstrap_sha256": hashlib.sha256((root / "bootstrap.py").read_bytes()).hexdigest(),
            "project_metadata_sha256": hashlib.sha256(
                (root / "pyproject.toml").read_bytes()
            ).hexdigest(),
        }
    )
    write_verification(root, prefix, device, conda, executable, environment, report)
    return report


def main():
    """Install or verify the selected requirements inside a dedicated Conda prefix."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prefix", type=Path, default=root / ".conda")
    parser.add_argument("--conda")
    parser.add_argument("--dev", action="store_true", help="Include requirements-dev.txt")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check an existing environment without installing",
    )
    args = parser.parse_args()
    try:
        result = install_environment(
            root,
            args.prefix,
            args.device,
            resolve_conda(args.conda),
            development=args.dev,
            verify_only=args.verify_only,
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
