"""Build a CRC-checked additive release without datasets, environments, or checkpoints."""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    """Package classification sources and actual verification reports under one additive subtree."""
    from cue_routing.io import read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Choose a new output path instead of overwriting a release")
    audit = read_json(ROOT / "classification/verification/audit.json")
    if audit.get("passed") is not True:
        raise ValueError("A passing source/evidence audit is required before packaging")
    sys.path.insert(0, str(ROOT / "classification"))
    from clvision.identity import source_hash

    if audit["source_sha256"] != source_hash(ROOT):
        raise ValueError("Sources changed after the release audit")
    paths = [
        path
        for path in (ROOT / "classification").rglob("*")
        if path.is_file()
        and not set(path.parts) & {"__pycache__", ".ruff_cache", ".cache", "data", "runs"}
        and path.suffix != ".pyc"
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(arguments.output) as archive:
        if archive.testzip() is not None or any(
            not name.startswith("classification/") for name in archive.namelist()
        ):
            raise ValueError("Release ZIP failed CRC or additive-path checks")
    print(
        {
            "archive": str(arguments.output.resolve()),
            "files": len(paths),
            "bytes": arguments.output.stat().st_size,
        }
    )


if __name__ == "__main__":
    main()
