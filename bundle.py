"""Package the portable project without virtual environments, datasets, caches, or runs."""

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


def main():
    """Write an atomic, CRC-checked source bundle with the actual verification reports."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root.parent / "cue_credit_routing.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in (".py", ".md", ".json", ".txt", ".toml", ".yml")
    ]
    paths += [root / ".gitignore"]
    for folder in (
        "src",
        "tests",
        "configs",
        "references",
        "verification",
        "classification",
        "office_home",
    ):
        paths += [
            path
            for path in (root / folder).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".md", ".json", ".txt")
            and not set(path.relative_to(root / folder).parts)
            & {"__pycache__", ".cache", ".ruff_cache", "data", "runs", "results", ".conda", ".git"}
            and not any(part.endswith(".egg-info") for part in path.parts)
        ]
    descriptor, temporary = tempfile.mkstemp(suffix=".zip", dir=args.output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(paths)):
                archive.write(path, str(Path(root.name) / path.relative_to(root)))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Bundle CRC verification failed")
            count = len(archive.infolist())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        {"bundle": str(args.output.resolve()), "files": count, "bytes": args.output.stat().st_size}
    )


if __name__ == "__main__":
    main()
