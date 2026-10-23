"""Build a verified full source release including the shared environment and experiment CLIs."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    """Require fresh classification evidence, then include every shared runtime dependency."""
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
    subprocess.run(
        [sys.executable, str(ROOT / "bundle.py"), "--output", str(arguments.output.resolve())],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
