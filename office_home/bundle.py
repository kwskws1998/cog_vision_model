"""Bundle the complete project or only the Office-Home addition without data or environments."""

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Write an atomic CRC-checked ZIP, with an add-on layout safe for the existing checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--addon-only", action="store_true")
    arguments = parser.parse_args()
    allowed = {".py", ".md", ".json", ".txt", ".toml", ".yml"}
    excluded = {".git", ".conda", ".cache", "data", "runs", "__pycache__", ".ruff_cache"}
    paths = []
    starts = [ROOT / "office_home"] if arguments.addon_only else [ROOT]
    for start in starts:
        for directory, folders, files in os.walk(start, followlinks=False):
            folders[:] = [
                name for name in folders if name not in excluded and not name.endswith(".egg-info")
            ]
            for name in files:
                path = Path(directory) / name
                if path.suffix in allowed or path.name == ".gitignore":
                    paths.append(path)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".zip", dir=arguments.output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(paths):
                relative = path.relative_to(ROOT)
                name = relative if arguments.addon_only else Path(ROOT.name) / relative
                handle.write(path, str(name))
        with zipfile.ZipFile(temporary) as handle:
            if handle.testzip() is not None:
                raise RuntimeError("ZIP CRC check failed")
        os.replace(temporary, arguments.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(
        {
            "bundle": str(arguments.output.resolve()),
            "files": len(paths),
            "bytes": arguments.output.stat().st_size,
            "addon_only": arguments.addon_only,
        }
    )


if __name__ == "__main__":
    main()
