"""Regression cases for defects identified during the full Conda and code review."""

import importlib.util
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_protocol import TinyClassifier

from cue_routing.cli import matrix, report_matrix
from cue_routing.config import DOMAINS, METHODS, ExperimentConfig
from cue_routing.data import load_manifest, make_fixture, safe_relative
from cue_routing.engine import evaluate_loader, train, vram_stats
from cue_routing.io import (
    atomic_json,
    atomic_torch,
    canonical_hash,
    capture_rng,
    environment,
    file_hash,
    read_json,
    restore_rng,
    run_lock,
    seed_everything,
    source_hash,
    state_hash,
)
from cue_routing.metrics import classification_metrics
from cue_routing.models import CueReadout
from cue_routing.protocol import record_target_access, seal_selection
from cue_routing.routing import route_gates

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reviewed_bootstrap", ROOT / "bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


def completed_matrix(directory, seeds=(0,)):
    """Fabricate explicitly artificial artifacts to exercise control flow, never efficacy."""
    directory = Path(directory)
    seed_everything(0)
    base = ExperimentConfig(device="cpu", precision="fp32")
    plan = [
        {"method": method, "target": domain, "seed": seed}
        for seed in sorted(seeds)
        for domain in DOMAINS
        for method in METHODS
    ]
    plan_contract = {
        "schema": 2,
        "base_config": base.to_dict(),
        "runs": plan,
        "dataset_id": "artificial-unit-test",
        "source_sha256": source_hash(ROOT),
    }
    atomic_json(directory / "matrix_plan.json", plan_contract)
    for specification in plan:
        run_dir = (
            directory
            / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
        )
        paths = [f"{specification['target']}/test-{index}.png" for index in range(3)]
        labels = torch.arange(3)
        contract = {
            "config": {**base.to_dict(), **specification},
            "dataset_id": plan_contract["dataset_id"],
            "source_sha256": plan_contract["source_sha256"],
            "initial_model_sha256": f"matched-{specification['seed']}",
            "runtime": environment(),
            "target_n": 3,
            "target_partition_sha256": canonical_hash(
                [[name, index] for index, name in enumerate(paths)]
            ),
        }
        atomic_json(run_dir / "contract.json", contract)
        atomic_torch(run_dir / "best.pt", {"unit_test_only": torch.zeros(1)})
        seal = seal_selection(run_dir, contract, {"step": 5000}, 5000)
        logits = torch.nn.functional.one_hot(labels, 7).float() * 3
        atomic_torch(
            run_dir / "target_predictions.pt",
            {
                "logits": logits,
                "targets": labels,
                "paths": paths,
                "contract_hash": canonical_hash(contract),
            },
        )
        atomic_json(
            run_dir / "target_metrics.json",
            {
                **specification,
                "benchmark_evidence": True,
                "metrics": classification_metrics(logits, labels),
                "selection": seal,
                "contract_hash": canonical_hash(contract),
                "prediction_sha256": file_hash(run_dir / "target_predictions.pt"),
            },
        )
        record_target_access(run_dir, seal)
    return base, plan_contract


class CondaReviewTests(unittest.TestCase):
    """Guard CPU/CUDA build selection and avoid accidental Conda base modifications."""

    def test_explicit_build_tags(self):
        cuda, index = BOOTSTRAP.wheel_specification("cuda", "Linux")
        self.assertEqual(cuda, ["torch==2.8.0+cu126", "torchvision==0.23.0+cu126"])
        self.assertTrue(index.endswith("/cu126"))
        cpu, _ = BOOTSTRAP.wheel_specification("cpu", "Linux")
        self.assertEqual(cpu[0], "torch==2.8.0+cpu")
        mac, _ = BOOTSTRAP.wheel_specification("cpu", "Darwin")
        self.assertEqual(mac[0], "torch==2.8.0")
        with self.assertRaises(ValueError):
            BOOTSTRAP.wheel_specification("cuda", "Darwin")

    def test_prefix_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            root.mkdir()
            for prefix in (root, root.parent):
                with self.assertRaises(ValueError):
                    BOOTSTRAP.validate_prefix(prefix, root)
            unrelated = Path(temporary) / "unrelated"
            unrelated.mkdir()
            with self.assertRaises(ValueError):
                BOOTSTRAP.validate_prefix(unrelated, root)
            (unrelated / "conda-meta").mkdir()
            (unrelated / "conda-meta" / "history").touch()
            with self.assertRaises(ValueError):
                BOOTSTRAP.validate_prefix(unrelated, root, base=unrelated)

    def test_audit_rejects_failed_smoke_with_matching_metadata(self):
        spec = importlib.util.spec_from_file_location("reviewed_audit", ROOT / "audit.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        report = {
            "benchmark_evidence": False,
            "matched_initial_weights": True,
            "environment": {"environment_manager": "conda"},
            "methods": {
                method: {"backbone_updated": True, "diagnostics": {"loss": 1.0}}
                for method in METHODS
            },
        }
        self.assertTrue(audit.valid_evidence("pretrained_cpu_smoke", report))
        report["methods"]["cue"]["backbone_updated"] = False
        self.assertFalse(audit.valid_evidence("pretrained_cpu_smoke", report))


class LifecycleReviewTests(unittest.TestCase):
    """Verify idempotent completed runs, pre-target matching, and interrupted commits."""

    def test_completed_evaluated_matrix_is_read_only_on_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, plan = completed_matrix(temporary)
            with (
                patch(
                    "cue_routing.cli.load_manifest",
                    return_value={"kind": "pacs", "dataset_id": plan["dataset_id"]},
                ),
                patch("cue_routing.cli.train") as training,
                patch("cue_routing.cli.evaluate_target") as evaluation,
            ):
                result = matrix(config, "mock-manifest", temporary, ROOT, [0], evaluate=True)
            training.assert_not_called()
            evaluation.assert_not_called()
            self.assertEqual(result["runs"], 12)

    def test_seed_input_order_is_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, plan = completed_matrix(temporary, seeds=(0, 2))
            with (
                patch(
                    "cue_routing.cli.load_manifest",
                    return_value={"kind": "pacs", "dataset_id": plan["dataset_id"]},
                ),
                patch("cue_routing.cli.train") as training,
            ):
                result = matrix(config, "mock-manifest", temporary, ROOT, [2, 0])
            training.assert_not_called()
            self.assertEqual(result["reused_completed_runs"], 24)

    def test_unmatched_initialization_rejected_before_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, plan = completed_matrix(temporary)
            affected = Path(temporary) / "sketch_seed0_cue"
            contract = read_json(affected / "contract.json")
            contract["initial_model_sha256"] = "different-initialization"
            atomic_json(affected / "contract.json", contract)
            seal_selection(affected, contract, {"step": 5000}, 5000)
            with (
                patch(
                    "cue_routing.cli.load_manifest",
                    return_value={"kind": "pacs", "dataset_id": plan["dataset_id"]},
                ),
                patch("cue_routing.cli.train") as training,
                patch("cue_routing.cli.evaluate_target") as evaluation,
            ):
                with self.assertRaisesRegex(ValueError, "matched comparison"):
                    matrix(config, "mock-manifest", temporary, ROOT, [0], evaluate=True)
            training.assert_not_called()
            evaluation.assert_not_called()

    def test_changed_dataset_rejected_before_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = completed_matrix(temporary)
            with (
                patch(
                    "cue_routing.cli.load_manifest",
                    return_value={"kind": "pacs", "dataset_id": "changed"},
                ),
                patch("cue_routing.cli.train") as training,
            ):
                with self.assertRaisesRegex(ValueError, "plan differs"):
                    matrix(config, "mock-manifest", temporary, ROOT, [0])
            training.assert_not_called()

    def test_prediction_paths_must_match_target_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed_matrix(temporary)
            affected = Path(temporary) / "sketch_seed0_cue"
            predictions = torch.load(affected / "target_predictions.pt", weights_only=True)
            predictions["paths"].reverse()
            atomic_torch(affected / "target_predictions.pt", predictions)
            metrics = read_json(affected / "target_metrics.json")
            metrics["prediction_sha256"] = file_hash(affected / "target_predictions.pt")
            atomic_json(affected / "target_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "held-out target partition"):
                report_matrix(temporary)

    def test_crash_before_first_checkpoint_restarts_deterministically(self):
        config = ExperimentConfig(
            steps=2,
            eval_every=2,
            image_size=64,
            batch_per_domain=1,
            eval_batch_size=7,
            device="cpu",
            precision="fp32",
            pretrained=False,
            workers=0,
            debug=True,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyClassifier(),
            ),
        ):
            root = Path(temporary)
            manifest = make_fixture(root / "images", None)
            train(config, manifest, root / "full", ROOT)
            train(config, manifest, root / "interrupted", ROOT)
            for name in ("last.pt", "history.json", "summary.json", "best.pt", "selection.json"):
                (root / "interrupted" / name).unlink()
            train(config, manifest, root / "interrupted", ROOT, resume=True)
            left = torch.load(root / "full" / "last.pt", weights_only=True)
            right = torch.load(root / "interrupted" / "last.pt", weights_only=True)
            self.assertEqual(state_hash(left["model"]), state_hash(right["model"]))

    def test_writer_lock_exclusion_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            with run_lock(temporary):
                with self.assertRaises(RuntimeError):
                    with run_lock(temporary):
                        self.fail("Concurrent writer entered the run")
            code = "import os,sys; from cue_routing.io import run_lock; guard=run_lock(sys.argv[1]); guard.__enter__(); os._exit(0)"
            subprocess.run([sys.executable, "-c", code, temporary], check=True)
            self.assertTrue((Path(temporary) / ".running.json").exists())
            with run_lock(temporary):
                self.assertTrue((Path(temporary) / ".running.json").exists())

    def test_python_numpy_and_torch_rng_restore(self):
        seed_everything(719)
        saved = capture_rng("cpu")
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.random()
        np.random.random()
        torch.rand(8)
        restore_rng(saved)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
        self.assertIsNone(saved["cuda"])


class InputReviewTests(unittest.TestCase):
    """Reject inputs that previously failed late or escaped manifest path constraints."""

    def test_public_gate_function_rejects_invalid_input(self):
        logits = torch.randn(2, 4, 7)
        for weights in (torch.ones(4), torch.full((4,), float("nan")), torch.ones(3) / 3):
            with self.assertRaises(ValueError):
                route_gates(logits, weights)
        with self.assertRaises(ValueError):
            route_gates(logits.half(), torch.ones(4).half() / 4)

    def test_invalid_token_grids_rejected(self):
        readout = CueReadout(8, 7)
        tokens = torch.randn(2, 49, 8)
        for grid in ((-7, -7), (7.0, 7), (7,), (0, 49)):
            with self.assertRaises(ValueError):
                readout(tokens, token_grid=grid)

    def test_noncanonical_paths_and_symlinked_class_rejected(self):
        with self.assertRaises(ValueError):
            safe_relative("photo//dog/image.png")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = make_fixture(root / "images", None)
            original = root / "images" / "photo" / "dog"
            moved = root / "moved-dog"
            original.rename(moved)
            original.symlink_to(moved, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlinked"):
                load_manifest(manifest_path, ROOT / "references" / "pacs_splits")

    def test_memory_budget_violation_is_explicit(self):
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=19 * 2**30),
            patch("torch.cuda.max_memory_reserved", return_value=21 * 2**30),
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                vram_stats(ExperimentConfig())

    def test_evaluation_enforces_memory_guard(self):
        model = TinyClassifier()
        batches = [(torch.randn(1, 3, 64, 64), torch.tensor([0]))]
        config = ExperimentConfig(device="cpu", precision="fp32")
        with patch("cue_routing.engine.vram_stats", side_effect=RuntimeError("memory-budget-test")):
            with self.assertRaisesRegex(RuntimeError, "memory-budget-test"):
                evaluate_loader(model, batches, config)
