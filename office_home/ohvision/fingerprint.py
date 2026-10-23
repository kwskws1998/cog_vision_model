"""Bind Office-Home runs to both the shared PACS core and this additive extension."""

from pathlib import Path

from cue_routing.io import canonical_hash, file_hash
from cue_routing.io import source_hash as shared_source_hash


def source_hash(project_root):
    """Exclude generated artifacts while fingerprinting all extension code and recipes."""
    root = Path(project_root)
    extension = root / "office_home"
    paths = [
        path
        for path in extension.rglob("*")
        if path.is_file()
        and path.suffix in (".py", ".json", ".toml", ".yml", ".txt")
        and not set(path.relative_to(extension).parts)
        & {"verification", "data", "runs", ".cache", ".ruff_cache", "__pycache__"}
    ]
    return canonical_hash(
        {
            "schema": 1,
            "shared_core_source_sha256": shared_source_hash(root),
            "office_home": {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)},
        }
    )
