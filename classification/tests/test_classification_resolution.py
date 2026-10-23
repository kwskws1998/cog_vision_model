"""ERM-only scope and resolution identity across real loaders, training, and sealed reports."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from clvision.ablation_settings import AblationSettings
from clvision.ablation_workflow import run_ablation
from clvision.config import Config
from clvision.cue_settings import CueSettings, ErmSettings
from clvision.cue_workflow import run_cue_study
from clvision.data import NORMALIZATION, ImageDataset, make_fixture, make_loader, transform_image
from clvision.identity import source_hash
from clvision.profile import memory_signature, verify_profile
from PIL import Image

from cue_routing.io import canonical_hash, environment, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]


class ResolutionTests(unittest.TestCase):
    """Prevent silent native-size evaluation or accidental CUE retraining in baseline jobs."""

    def setUp(self):
        """Use isolated artifacts and real small CPU fixtures."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        """Remove only this test's files."""
        self.temporary.cleanup()

    def test_cli_scope_and_resolution_for_both_entrypoints(self):
        """Both public entrypoints resolve the intended budgets without touching data."""
        for entry in ("run.py", "experiment.py"):
            for args, mode in (
                (["erm"], "erm_only"),
                (
                    [
                        "ablation",
                        "--config",
                        str(ROOT / "classification/configs/tiny32_ablation_5090.json"),
                    ],
                    "matched_ablation",
                ),
            ):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "classification" / entry), *args, "--dry-run"],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                value = json.loads(result.stdout)
                self.assertEqual(
                    (value["mode"], value["training_runs"], value["development_runs"]), (mode, 4, 0)
                )
                self.assertEqual(
                    value["input_shapes"]["tiny_imagenet"],
                    [64, 64] if mode == "erm_only" else [32, 32],
                )
                if mode == "erm_only":
                    self.assertEqual(value["methods"], ["erm"])
                    self.assertEqual(value["input_shapes"]["cifar100"], [32, 32])

    def test_erm_changes_only_objective_and_rejects_invalid_size(self):
        """Baseline controls share every CUE training setting and cannot accept invalid resolutions."""
        value = read_json(ROOT / "classification/configs/erm_5090.json")
        erm, cue = ErmSettings.from_dict(value), CueSettings.from_dict(value)
        for dataset in erm.datasets:
            self.assertEqual(erm.recipe(dataset), replace(cue.recipe(dataset), method="erm"))
        for size in (0, 16, 33, 128, True, 32.0, "32"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Config(tiny_imagenet_size=size).validate()

    def test_resize_before_augmentation_and_normalization(self):
        """Verify actual pixel values against explicit bilinear downsampling and keyed crops."""
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        resized = np.asarray(Image.fromarray(image).resize((32, 32), Image.Resampling.BILINEAR))
        for training in (False, True):
            expected = resized.copy()
            if training:
                rng = np.random.default_rng(823)
                y, x = rng.integers(0, 9, size=2)
                expected = np.pad(expected, ((4, 4), (4, 4), (0, 0)))[y : y + 32, x : x + 32]
                if rng.random() < 0.5:
                    expected = expected[:, ::-1]
            expected = torch.tensor(expected.copy()).permute(2, 0, 1).float() / 255
            mean, std = NORMALIZATION["tiny_imagenet"]
            expected = (expected - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[
                :, None, None
            ]
            actual = transform_image(image, "tiny_imagenet", training, 823, tiny_imagenet_size=32)
            self.assertTrue(torch.equal(actual, expected))

    def test_native_processing_matches_previous_commit(self):
        """Golden tensors from commit 1e7036a protect the already completed native-size recipe."""
        digests = {
            ("cifar100", False): "e33eea14ff28797412cda0c363edb2b6e2587d9d58ec30a465c42660ebec1f62",
            ("cifar100", True): "92f922eb9cb567e017f7463ab345b66555a37c34cdbf735f9e0bd0fbd0d9fb22",
            (
                "tiny_imagenet",
                False,
            ): "39a9fa32d31fbead151fefde357219063089c4d0269fb398321e0a4a43226222",
            (
                "tiny_imagenet",
                True,
            ): "c34958e1cd5ff2195d5c871b998f731350d37cefe40153eba6b396519b3b6ad8",
        }
        for (dataset, training), expected in digests.items():
            size = 32 if dataset == "cifar100" else 64
            image = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
            result = transform_image(image, dataset, training, 823)
            self.assertEqual(hashlib.sha256(result.numpy().tobytes()).hexdigest(), expected)
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        self.assertTrue(
            torch.equal(
                transform_image(image, "cifar100", True, 823),
                transform_image(image, "cifar100", True, 823, tiny_imagenet_size=32),
            )
        )

    def test_all_splits_preserve_identities_and_workers_preserve_resize(self):
        """Resizing changes tensors, never examples, targets, manifests, or worker determinism."""
        manifest = make_fixture(self.folder, "tiny_imagenet")
        before = canonical_hash(manifest)
        config = ErmSettings.from_dict(
            {"datasets": ["tiny_imagenet"], "debug": True, "training": {"tiny_imagenet_size": 32}}
        ).recipe("tiny_imagenet")
        for split in ("train", "validation", "test"):
            small = ImageDataset(manifest, split, tiny_imagenet_size=32)
            native = ImageDataset(manifest, split)
            for index in range(len(small)):
                self.assertEqual(small[index][0].shape, (3, 32, 32))
                self.assertEqual(native[index][0].shape, (3, 64, 64))
                self.assertEqual(small.sample_id(index), native.sample_id(index))
                self.assertEqual(small[index][1], native[index][1])
            zero = list(make_loader(manifest, config, split, epoch=3))
            two = list(make_loader(manifest, replace(config, workers=2), split, epoch=3))
            self.assertEqual(len(zero), len(two))
            for (images, labels), (other_images, other_labels) in zip(zero, two):
                self.assertEqual(tuple(images.shape[-2:]), (32, 32))
                self.assertTrue(torch.equal(images, other_images))
                self.assertTrue(torch.equal(labels, other_labels))
        self.assertEqual(canonical_hash(manifest), before)

    def test_jpeg_record_path_also_resizes(self):
        """Production Tiny uses image records instead of fixture arrays and must follow the same path."""
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        Image.fromarray(image).save(self.folder / "example.JPEG")
        manifest = {
            "root": str(self.folder),
            "dataset": "tiny_imagenet",
            "splits": {"train": [0], "validation": [0], "test": [0]},
            "records": [["example.JPEG", 17]],
            "test_records": [["example.JPEG", 17]],
        }
        with Image.open(self.folder / "example.JPEG") as source:
            expected = transform_image(
                np.array(source.convert("RGB")), "tiny_imagenet", tiny_imagenet_size=32
            )
        for split in ("validation", "test"):
            image, label = ImageDataset(manifest, split, tiny_imagenet_size=32)[0]
            self.assertEqual(label, 17)
            self.assertTrue(torch.equal(image, expected))
        self.assertEqual(
            ImageDataset(manifest, "train", tiny_imagenet_size=32)[0][0].shape, (3, 32, 32)
        )

    def test_profile_rejects_different_resolution(self):
        """A successful 32px memory measurement cannot authorize a 64px workload."""
        config = Config(dataset="tiny_imagenet", tiny_imagenet_size=32)
        evidence = {
            "measured": True,
            "passes_budget": True,
            "source_sha256": source_hash(ROOT),
            "dataset_id": "fixture",
            "environment": environment(),
            "optimizer_steps_per_method": 10,
            "methods": {"cue": {"signature": memory_signature(config)}},
        }
        verify_profile(evidence, ROOT, {"dataset_id": "fixture"}, config, methods=("cue",))
        with self.assertRaisesRegex(ValueError, "signature differs"):
            verify_profile(
                evidence,
                ROOT,
                {"dataset_id": "fixture"},
                replace(config, tiny_imagenet_size=64),
                methods=("cue",),
            )

    def test_erm_execution_reports_only_baseline_and_reuses_completion(self):
        """Run real ERM train/test on both datasets and reject switching objective on resume."""
        data, directory = self.folder / "data", self.folder / "erm"
        settings = ErmSettings.from_dict({"debug": True, "seeds": [0]})
        for name in settings.datasets:
            make_fixture(data, name)
        from clvision.cue_workflow import evaluate_run

        def guarded(manifest, run_dir, root, seal):
            """Require all baseline training completions before final evaluation."""
            self.assertEqual(len(seal["runs"]), 2)
            self.assertEqual(len(list(directory.rglob("completion.json"))), 2)
            return evaluate_run(manifest, run_dir, root, seal)

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("clvision.cue_workflow.evaluate_run", side_effect=guarded),
        ):
            report = run_cue_study(ROOT, data, directory, settings)
            hashes = {str(path): file_hash(path) for path in directory.rglob("completion.json")}
            self.assertEqual(report, run_cue_study(ROOT, data, directory, settings))
            self.assertEqual(
                hashes, {str(path): file_hash(path) for path in directory.rglob("completion.json")}
            )
            with self.assertRaisesRegex(ValueError, "scope or source"):
                run_cue_study(ROOT, data, directory, CueSettings.from_dict(settings.to_dict()))
        self.assertEqual(report["mode"], "erm_only")
        self.assertFalse(report["benchmark_evidence"])
        for dataset, size in (("cifar100", 32), ("tiny_imagenet", 64)):
            item = report["datasets"][dataset]
            self.assertNotIn("cue_measured", item)
            self.assertEqual(item["erm_measured"]["input_size"], size)
            self.assertEqual(item["erm_measured"]["config"]["method"], "erm")
        progress = read_json(directory / "progress.json")
        self.assertEqual(
            (
                progress["status"],
                progress["total_training_runs"],
                progress["estimated_remaining_seconds"],
            ),
            ("complete", 2, 0),
        )

    def test_tiny32_matched_training_report_and_resolution_resume_guard(self):
        """Exercise genuine 32px ERM/CUE training, sealed tests, report sizes, and identity checks."""
        value = read_json(ROOT / "classification/configs/tiny32_ablation_5090.json")
        settings = AblationSettings.from_dict({**value, "debug": True, "seeds": [0]})
        data, directory = self.folder / "data", self.folder / "tiny32"
        make_fixture(data, "tiny_imagenet")
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_ablation(ROOT, data, directory, settings)
            self.assertEqual(report, run_ablation(ROOT, data, directory, settings))
            changed = AblationSettings.from_dict(
                {**settings.to_dict(), "training": {**settings.training, "tiny_imagenet_size": 64}}
            )
            with self.assertRaisesRegex(ValueError, "scope or source"):
                run_ablation(ROOT, data, directory, changed)
        item = report["datasets"]["tiny_imagenet"]
        self.assertEqual(item["input_size"], 32)
        self.assertEqual(set(item["methods"]), {"erm", "cue"})
        self.assertEqual(item["selection"]["candidates"], [])
        self.assertIn("cue_minus_erm_pp", item["paired_differences"])
        self.assertIn("Model input: 32 x 32", (directory / "comparison.md").read_text())
        self.assertNotIn("our 64px", (directory / "comparison.md").read_text())
