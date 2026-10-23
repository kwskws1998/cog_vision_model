"""Run Office-Home experiments with the existing project-local Conda environment."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "office_home"))

from ohvision.cli import main

if __name__ == "__main__":
    main(ROOT)
