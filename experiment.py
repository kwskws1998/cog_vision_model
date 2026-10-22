"""Standalone entrypoint; all generated artifacts remain inside this project."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cue_routing.cli import main

if __name__ == "__main__":
    main(ROOT)
