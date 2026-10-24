"""Check CLI-controlled recipes, bounded searches, and real synthetic execution through both drivers."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clvision.tuning_cli import add_tuning_arguments, resolve_tuning_arguments

from cue_routing.io import file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "classification/references/cifar100_seed0_cue_temperature005.json"
PRODUCTION = [
    "--dataset",
    "cifar100",
    "--seeds",
    "0",
    "--activation",
    "preact",
    "--initialization",
    "msra",
    "--normalization",
    "zipfls",
    "--epochs",
    "200",
    "--milestones",
    "100",
    "150",
    "--batch-size",
    "128",
    "--eval-batch-size",
    "128",
    "--learning-rate",
    "0.1",
    "--warmup-epochs",
    "0",
    "--momentum",
    "0.9",
    "--weight-decay",
    "1e-4",
    "--lr-decay",
    "0.1",
    "--gate-tau",
    "0.3",
    "--gate-temperature",
    "0.05",
    "--joint-floor",
    "0.5",
    "--gradient-clip",
    "0",
    "--precision",
    "fp32",
    "--eval-precision",
    "fp32",
    "--device",
    "cuda",
    "--workers",
    "4",
    "--max-vram-gib",
    "28",
    "--tune-gate-tau",
    "0.3",
    "--tune-gate-temperature",
    "0.025",
    "0.1",
    "--tune-joint-floor",
    "0.5",
    "--search-strategy",
    "grid",
    "--search-seed",
    "2027",
    "--max-trials",
    "2",
    "--selection-metric",
    "accuracy",
    "--no-include-baseline",
    "--stages",
    "preflight",
    "tune",
    "evaluate",
    "--preflight-steps",
    "10",
    "--reserve-minutes",
    "10",
    "--time-safety-factor",
    "1.25",
    "--progress",
    "plain",
    "--logging-steps",
    "200",
    "--progress-interval",
    "30",
]


class TuningCliTests(unittest.TestCase):
    """Guard against silently ignoring explicit flags or changing the number of training jobs."""

    def setUp(self):
        """Keep plans, fixtures, and candidate artifacts outside executable source directories."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.parser = argparse.ArgumentParser()
        add_tuning_arguments(self.parser, ROOT)

    def resolve(self, tokens):
        """Resolve an isolated command without creating any run or touching a dataset."""
        arguments = self.parser.parse_args(["--runs-dir", str(self.folder / "run"), *tokens])
        return resolve_tuning_arguments(arguments, ROOT)

    def invoke(self, entrypoint, tokens):
        """Use the same Conda interpreter and clean import path as the production launcher."""
        environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "classification" / entrypoint),
                "tune",
                "--runs-dir",
                str(self.folder / "run"),
                *tokens,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_production_flags_match_reference_without_data_access(self):
        """The documented command schedules exactly two additional seed-zero CUE candidates."""
        with patch("clvision.data.read_array", side_effect=AssertionError("No data in preview")):
            runs, stages, preview = self.resolve(PRODUCTION + ["--reference", str(REFERENCE)])
        self.assertEqual(len(runs), 1)
        self.assertEqual(stages, ("preflight", "tune", "evaluate"))
        self.assertEqual(preview["maximum_training_runs"], 2)
        self.assertFalse(runs[0].include_baseline)
        self.assertEqual(runs[0].reference["validation"]["accuracy"], 0.7494)
        self.assertEqual([c.gate_temperature for c in runs[0].candidates], [0.025, 0.1])
        self.assertEqual(runs[0].config.to_dict(), read_json(REFERENCE)["config"])
        self.assertFalse((self.folder / "run").exists())

    def test_arbitrary_common_settings_and_baseline_order(self):
        """Non-example inputs reach every candidate, with no fixed paper recipe or deadline."""
        runs, stages, preview = self.resolve(
            [
                "--dataset",
                "tiny_imagenet",
                "--seeds",
                "3",
                "9",
                "--activation",
                "postact",
                "--tiny-imagenet-size",
                "32",
                "--epochs",
                "17",
                "--milestones",
                "9",
                "14",
                "--batch-size",
                "256",
                "--eval-batch-size",
                "512",
                "--learning-rate",
                "0.23",
                "--warmup-epochs",
                "3",
                "--momentum",
                "0.85",
                "--weight-decay",
                "0.002",
                "--lr-decay",
                "0.2",
                "--gradient-clip",
                "1.2",
                "--precision",
                "bf16",
                "--eval-precision",
                "fp16",
                "--gate-temperature",
                "0.14",
                "--joint-floor",
                "0.24",
                "--tune-gate-tau",
                "0.1",
                "0.2",
                "--workers",
                "8",
                "--max-vram-gib",
                "25",
                "--include-baseline",
                "--stages",
                "preflight",
                "baseline",
                "tune",
                "evaluate",
            ]
        )
        self.assertEqual(preview["maximum_training_runs"], 6)
        self.assertEqual(stages[1], "baseline")
        for seed, run in zip((3, 9), runs):
            self.assertEqual(run.directory.name, f"seed{seed}")
            self.assertEqual(run.config.method, "erm")
            self.assertIsNone(run.deadline)
            for candidate in run.candidates:
                self.assertEqual(candidate.seed, seed)
                self.assertEqual(candidate.learning_rate, 0.23)
                self.assertEqual(candidate.batch_size, 256)
                self.assertEqual(candidate.eval_batch_size, 512)
                self.assertEqual(candidate.warmup_epochs, 3)
                self.assertEqual(candidate.milestones, (9, 14))
                self.assertEqual(candidate.gate_temperature, 0.14)
                self.assertEqual(candidate.joint_floor, 0.24)
                self.assertEqual(candidate.workers, 8)
                self.assertEqual(candidate.precision, "bf16")

    def test_random_grid_is_bounded_and_reproducible(self):
        """Random search only permutes the declared finite grid and retains training seeds."""
        tokens = [
            "--tune-gate-tau",
            "0.1",
            "0.2",
            "0.3",
            "--tune-joint-floor",
            "0",
            "0.5",
            "--search-strategy",
            "random",
            "--search-seed",
            "7",
            "--max-trials",
            "3",
        ]
        runs, _, first = self.resolve(tokens)
        _, _, second = self.resolve(tokens)
        self.assertEqual(first, second)
        self.assertEqual(first["grid_size"], 6)
        self.assertEqual(first["maximum_training_runs"], 3)
        self.assertEqual({config.seed for config in runs[0].candidates}, {0})
        self.assertEqual(len(set(runs[0].candidates)), 3)

    def test_invalid_inputs_fail_before_outputs(self):
        """Reject mismatched references, duplicated grids, unsafe order, and invalid timing."""
        invalid = [
            ["--seeds", "0", "0"],
            ["--seeds", "-1"],
            ["--max-trials", "0"],
            ["--tune-gate-temperature", "0.1", "0.1"],
            ["--tune-gate-temperature", "0"],
            ["--tune-gate-tau", "nan"],
            ["--finish-by", "2026-10-25T10:00:00"],
            ["--cpu-threads", "0"],
            ["--preflight-steps", "1"],
            ["--stages", "tune", "evaluate"],
            ["--reference", str(REFERENCE)],
            ["--reference", str(REFERENCE), "--seeds", "0", "1"],
            ["--reference", str(REFERENCE), "--smoke"],
            PRODUCTION + ["--reference", str(REFERENCE), "--learning-rate", "0.2"],
        ]
        for tokens in invalid:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.resolve(tokens)
        self.assertFalse((self.folder / "run").exists())

    def test_entrypoints_share_full_production_preview(self):
        """Both public commands parse the exact user-facing recipe without importing an example."""
        tokens = PRODUCTION + ["--reference", str(REFERENCE), "--dry-run"]
        outputs = []
        for entrypoint in ("run.py", "experiment.py"):
            result = self.invoke(entrypoint, tokens)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(json.loads(result.stdout))
        self.assertEqual(*outputs)
        self.assertFalse((self.folder / "run").exists())

    def test_real_cli_smoke_resume_and_changed_recipe_rejected(self):
        """Run two CUE trials, test only the selected checkpoint, and reuse it on resume."""
        tokens = PRODUCTION + ["--smoke", "--progress", "off"]
        first = self.invoke("run.py", tokens)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        directory = self.folder / "run"
        result = read_json(directory / "comparison.json")
        self.assertEqual(result["protocol"]["maximum_training_runs"], 2)
        self.assertEqual(set(result["test_metrics"]), {"cue"})
        self.assertFalse((directory / "main/seed0_erm").exists())
        self.assertFalse(list((directory / "development").rglob("test_access.json")))
        predictions = directory / "main/seed0_cue/test_predictions.pt"
        before = file_hash(predictions)
        resumed = self.invoke("run.py", tokens + ["--resume"])
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(file_hash(predictions), before)
        self.assertEqual(len(read_json(directory / "main/seed0_cue/test_access.json")), 1)
        invocations = [
            json.loads(line) for line in (directory / "invocations.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(invocations), 2)
        self.assertEqual(invocations[0]["arguments"]["tune_gate_temperature"], [0.025, 0.1])
        changed = self.invoke("run.py", tokens + ["--resume", "--max-trials", "1"])
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("Study scope or source changed", changed.stderr)


if __name__ == "__main__":
    unittest.main()
