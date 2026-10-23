"""Bind selectable backbones to profiling, checkpoints, and independently verified reports."""

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.ablation_settings import AblationSettings
from clvision.ablation_workflow import run_ablation
from clvision.config import Config
from clvision.cue_settings import CueSettings
from clvision.cue_workflow import run_cue_study
from clvision.data import make_fixture
from clvision.engine import load_final, train, verify_run
from clvision.identity import source_hash
from clvision.profile import memory_signature, verify_profile
from clvision.workflow import report
from test_classification_training import debug_config, equal_states

from cue_routing.io import read_json

ROOT = Path(__file__).resolve().parents[2]


class ActivationWorkflowTests(unittest.TestCase):
    """Exercise real CPU training while reserving CUDA performance claims for server preflight."""

    def setUp(self):
        """Allocate independent data and checkpoint directories for each regression."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        """Delete only the current test's temporary artifacts."""
        self.temporary.cleanup()

    def test_preflight_cannot_be_reused_for_another_activation(self):
        """Reject another architecture even when source, device, precision, and batch agree."""
        config = Config()
        value = {
            "measured": True,
            "passes_budget": True,
            "source_sha256": source_hash(ROOT),
            "dataset_id": "fixture",
            "environment": {"gpu": "test"},
            "optimizer_steps_per_method": 10,
            "methods": {"cue": {"signature": memory_signature(config)}},
        }
        with patch("clvision.profile.environment", return_value=value["environment"]):
            verify_profile(value, ROOT, {"dataset_id": "fixture"}, config, methods=("cue",))
            with self.assertRaisesRegex(ValueError, "signature"):
                verify_profile(
                    value,
                    ROOT,
                    {"dataset_id": "fixture"},
                    replace(config, activation="preact"),
                    methods=("cue",),
                )

    def test_full_report_rejects_mixed_backbones_before_artifact_access(self):
        """A comparison cannot pool two architectures under one dataset/method label."""
        config = debug_config()
        specs = [
            (self.folder / "post", config, "selection", "main"),
            (self.folder / "pre", replace(config, activation="preact"), "selection", "main"),
        ]
        with patch("clvision.workflow.verify_run", side_effect=AssertionError("artifact access")):
            with self.assertRaisesRegex(ValueError, "one matched backbone activation"):
                report(self.folder, ROOT, specs, {}, {"benchmark_evidence": False})

    def test_preact_exact_resume_and_final_reconstruction(self):
        """Reconstruct the selected model and replay momentum, RNG, and updates exactly."""
        manifest = make_fixture(self.folder / "data")
        config = replace(debug_config(), activation="preact")
        full, resumed = self.folder / "full", self.folder / "resumed"
        with contextlib.redirect_stdout(io.StringIO()):
            train(config, manifest, full, ROOT, phase="debug")
            train(config, manifest, resumed, ROOT, phase="debug", stop_after=1)
            with self.assertRaisesRegex(ValueError, "Resume requires identical"):
                train(
                    replace(config, activation="postact"),
                    manifest,
                    resumed,
                    ROOT,
                    phase="debug",
                    resume=True,
                )
            train(config, manifest, resumed, ROOT, phase="debug", resume=True)
        first = torch.load(full / "last.pt", weights_only=True)
        second = torch.load(resumed / "last.pt", weights_only=True)
        for key in ("model", "optimizer", "scaler", "rng", "steps", "epoch"):
            self.assertTrue(equal_states(first[key], second[key]), key)
        model, loaded, contract, _ = load_final(resumed, ROOT, manifest)
        self.assertEqual(loaded.activation, "preact")
        self.assertIn("pre-activation", contract["backbone"])
        self.assertTrue(equal_states(model.state_dict(), second["model"]))
        self.assertEqual(read_json(resumed / "runtime_status.json")["activation"], "preact")
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            verify_run(resumed, ROOT, replace(config, activation="postact"))

    def test_reports_name_each_selected_backbone_and_reject_scope_changes(self):
        """Run both dataset architectures and ensure report prose follows the measured recipe."""
        data, directory = self.folder / "data", self.folder / "study"
        for dataset in ("cifar100", "tiny_imagenet"):
            make_fixture(data, dataset)
        settings = CueSettings.from_dict(
            {
                "seeds": [0],
                "debug": True,
                "training": {"activation": "preact"},
                "dataset_overrides": {"tiny_imagenet": {"activation": "postact"}},
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_cue_study(ROOT, data, directory, settings)
        for dataset, activation in (("cifar100", "preact"), ("tiny_imagenet", "postact")):
            report = result["datasets"][dataset]
            measured = report["cue_measured"]
            self.assertEqual(measured["activation"], activation)
            self.assertEqual(measured["config"]["activation"], activation)
            self.assertEqual("differs" in report["protocol_notes"][0], activation == "postact")
            self.assertIn(measured["backbone"], (directory / "comparison.md").read_text())
        changed = CueSettings.from_dict(
            {**settings.to_dict(), "training": {"activation": "postact"}}
        )
        with self.assertRaisesRegex(ValueError, "scope or source"):
            run_cue_study(ROOT, data, directory, changed)

    def test_preact_ablation_preserves_matched_initialization_and_report(self):
        """ERM and CUE use the same PreAct weights and selected backbone in a sealed comparison."""
        data, directory = self.folder / "data", self.folder / "ablation"
        make_fixture(data, "cifar100")
        settings = AblationSettings.from_dict(
            {"debug": True, "tune": False, "seeds": [0], "training": {"activation": "preact"}}
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_ablation(ROOT, data, directory, settings)
        row = result["datasets"]["cifar100"]
        self.assertEqual(row["activation"], "preact")
        self.assertIn("pre-activation", row["backbone"])
        self.assertFalse(row["published_comparison_is_matched"])
        self.assertEqual(set(row["methods"]), {"erm", "cue"})
        contracts = [
            read_json(directory / "main/cifar100" / f"seed0_{method}" / "contract.json")
            for method in ("erm", "cue")
        ]
        self.assertEqual(
            contracts[0]["initial_shared_model_sha256"],
            contracts[1]["initial_shared_model_sha256"],
        )
        self.assertIn(row["backbone"], (directory / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
