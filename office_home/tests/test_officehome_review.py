"""Verification semantics and preservation of the active PACS source contract."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohvision.config import METHODS

ROOT = Path(__file__).resolve().parents[2]


class OfficeHomeReviewTests(unittest.TestCase):
    """Reject favorable metadata when the recorded real-source optimizer path failed."""

    def test_audit_rejects_failed_real_source_update(self):
        specification = importlib.util.spec_from_file_location(
            "officehome_audit_review", ROOT / "office_home/audit.py"
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        report = {
            "benchmark_evidence": False,
            "matched_initial_weights": True,
            "matched_training_images_and_augmentations": True,
            "environment": {"environment_manager": "conda"},
            "verified_image_count": 15588,
            "target_loader_accessed": False,
            "optimizer_steps_per_method": 1,
            "methods": {
                method: {
                    "backbone_updated": True,
                    "diagnostics": {"loss": 1.0},
                    "optimizer_steps": 1,
                    "source_validation_domains_checked": ["Clipart", "Product", "Real_World"],
                }
                for method in METHODS
            },
        }
        self.assertTrue(module.valid_report("real_data_cpu_smoke", report))
        report["methods"]["cue"]["optimizer_steps"] = 0
        self.assertFalse(module.valid_report("real_data_cpu_smoke", report))

    def test_driver_stops_after_failed_gpu_profile(self):
        specification = importlib.util.spec_from_file_location(
            "officehome_driver_review", ROOT / "office_home/run.py"
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text("{}")
            with (
                patch("sys.argv", ["run.py", "train", "--manifest", str(manifest)]),
                patch.object(
                    module.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, ["profile"]),
                ) as execution,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.main()
            self.assertEqual(execution.call_count, 1)
            self.assertIn("profile", execution.call_args.args[0])
