"""Create a project-local Conda environment with explicit CPU or CUDA 12.6 wheels."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def resolve_conda(value=None):
    """Locate Conda without installing into or modifying the base environment."""
    candidate = value or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError(
            "Conda was not found; install Miniconda/Anaconda or pass --conda /absolute/path/to/conda"
        )
    return str(Path(candidate).resolve())


def wheel_specification(device, system):
    """Pin the local build tag so an existing CPU install cannot satisfy a CUDA setup."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and system != "Linux":
        raise ValueError("The CUDA benchmark environment is Linux; use --device cpu on this host")
    build = "+cu126" if device == "cuda" else ("+cpu" if system == "Linux" else "")
    index = (
        "https://download.pytorch.org/whl/cu126"
        if device == "cuda"
        else "https://download.pytorch.org/whl/cpu"
    )
    return [f"torch==2.8.0{build}", f"torchvision==0.23.0{build}"], index


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


def install_environment(root, prefix, device, conda):
    """Create Conda Python first, then install and verify official PyTorch wheel builds."""
    root = Path(root).resolve()
    packages, index = wheel_specification(device, platform.system())
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
    executable = prefix / ("python.exe" if platform.system() == "Windows" else "bin/python")
    subprocess.run(
        [
            str(executable),
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 11), 'The recipe requires Conda Python 3.11'",
        ],
        check=True,
        env=environment,
    )
    subprocess.run(
        [str(executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=True,
        env=environment,
    )
    subprocess.run(
        [str(executable), "-m", "pip", "install", *packages, "--index-url", index],
        check=True,
        env=environment,
    )
    subprocess.run(
        [str(executable), "-m", "pip", "install", "-e", str(root)], check=True, env=environment
    )
    subprocess.run([str(executable), "-m", "pip", "check"], check=True, env=environment)
    probe = "import json,sys,torch,torchvision; from pathlib import Path; p=Path(sys.prefix).resolve(); assert (p/'conda-meta'/'history').is_file(); assert Path(torch.__file__).resolve().is_relative_to(p); assert Path(torchvision.__file__).resolve().is_relative_to(p); print(json.dumps({'prefix':str(p),'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}))"
    result = subprocess.run(
        [str(executable), "-c", probe], check=True, env=environment, capture_output=True, text=True
    )
    report = json.loads(result.stdout)
    if (device == "cuda" and report["cuda_runtime"] != "12.6") or (
        device == "cpu" and report["cuda_runtime"] is not None
    ):
        raise RuntimeError("Installed PyTorch build does not match the requested device")
    report.update(
        {
            "environment_manager": "conda",
            "requested_device": device,
            "wheel_index": index,
            "environment_yml_sha256": hashlib.sha256(
                (root / "environment.yml").read_bytes()
            ).hexdigest(),
            "bootstrap_sha256": hashlib.sha256((root / "bootstrap.py").read_bytes()).hexdigest(),
            "project_metadata_sha256": hashlib.sha256(
                (root / "pyproject.toml").read_bytes()
            ).hexdigest(),
        }
    )
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
    return report


def main():
    """Install the selected build inside a dedicated Conda prefix without activation."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prefix", type=Path, default=root / ".conda")
    parser.add_argument("--conda")
    args = parser.parse_args()
    try:
        result = install_environment(root, args.prefix, args.device, resolve_conda(args.conda))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
