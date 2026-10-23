"""Protect reproducible preflight verification when entering from a fresh Python process."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.config import Config
from clvision.data import make_fixture
from clvision.experiment import Comparison
from clvision.experiment_settings import Search

from cue_routing.io import environment, seed_everything

ROOT = Path(__file__).resolve().parents[2]


class ExperimentRuntimeTests(unittest.TestCase):
    """Exercise environment normalization without requiring a CUDA device or full training."""

    def test_preflight_normalizes_environment_before_cached_profile_verification(self):
        """A fresh process must reproduce stored deterministic/TF32 flags before comparing profiles."""
        original = (
            torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
        )

        def restore():
            """Restore global PyTorch modes so this regression check cannot affect other tests."""
            torch.use_deterministic_algorithms(original[0])
            torch.backends.cudnn.benchmark = original[1]
            torch.backends.cudnn.deterministic = original[2]
            torch.backends.cudnn.allow_tf32 = original[3]
            torch.backends.cuda.matmul.allow_tf32 = original[4]

        self.addCleanup(restore)
        seed_everything(0, "cpu")
        cached_environment = environment()
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        def verify_cached_environment(root, directory, manifest, config, candidates, steps):
            """Represent the strict cached-profile environment comparison without GPU kernels."""
            self.assertEqual(environment(), cached_environment)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            return None, [None] * len(candidates)

        config = Config(
            method="erm",
            epochs=2,
            milestones=(1,),
            batch_size=2,
            eval_batch_size=4,
            workers=0,
            debug=True,
            debug_max_batches=1,
            device="cpu",
            precision="fp32",
            eval_precision="fp32",
        )
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temporary)
            make_fixture(folder / "data")
            with (
                patch(
                    "clvision.experiment.preflight_search",
                    side_effect=verify_cached_environment,
                ) as profile,
                Comparison(
                    root=ROOT,
                    data_root=folder / "data",
                    directory=folder / "run",
                    config=config,
                    search=Search([{}]),
                ) as comparison,
            ):
                comparison.preflight()
                self.assertTrue(comparison.ready)
                profile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
