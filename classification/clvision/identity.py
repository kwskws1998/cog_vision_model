"""Bind new results to immutable code and to the unchanged legacy experiment core."""

from pathlib import Path

from cue_routing.io import canonical_hash, file_hash
from cue_routing.io import source_hash as legacy_source_hash


def source_hash(root):
    """Exclude generated reports, data, checkpoints, and documentation from executable identity."""
    root = Path(root)
    extension = root / "classification"
    paths = [
        path
        for path in extension.rglob("*")
        if path.is_file()
        and path.suffix in (".py", ".json", ".txt", ".yml", ".toml")
        and not set(path.relative_to(extension).parts)
        & {"verification", "__pycache__", ".cache", ".ruff_cache", "data", "runs"}
    ]
    return canonical_hash(
        {
            "schema": 1,
            "legacy_core": legacy_source_hash(root),
            "classification": {
                str(path.relative_to(root)): file_hash(path) for path in sorted(paths)
            },
        }
    )
