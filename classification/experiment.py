"""Run the additive classification experiments inside the existing project Conda environment."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification"), str(ROOT / "office_home")]


def main():
    """Import PyTorch only after the deterministic CUDA environment is configured."""
    from clvision.cli import main as run

    run(ROOT)


if __name__ == "__main__":
    main()
