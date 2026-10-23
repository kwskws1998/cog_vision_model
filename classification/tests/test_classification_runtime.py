"""Regression checks for active-job identity, separated host waits, and deferred scalar reporting."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from clvision.config import Config
from clvision.losses import consistency_loss
from clvision.runtime import diagnostic_scalars, reset_run_memory, runtime_status, timed_batches

from cue_routing.io import read_json


class RuntimeTests(unittest.TestCase):
    """Check diagnostic semantics without requiring a CUDA device."""

    def test_run_memory_baseline_excludes_prior_inactive_cache(self):
        """Resetting a peak alone retains the prior cache; a new run must release it first."""
        allocator = {"reserved": 23, "peak": 23}
        events = []

        def collect():
            """Record collection before inactive memory is returned to the allocator."""
            events.append("collect")

        def release():
            """Retain live allocations while releasing a previous large profile's cached blocks."""
            events.append("release")
            allocator["reserved"] = 1

        def reset():
            """Model CUDA's peak-reset semantics: reset to current reservation, not zero."""
            events.append("reset")
            allocator["peak"] = allocator["reserved"]

        with (
            patch("clvision.runtime.gc.collect", side_effect=collect),
            patch("clvision.runtime.torch.cuda.empty_cache", side_effect=release),
            patch("clvision.runtime.torch.cuda.reset_peak_memory_stats", side_effect=reset),
        ):
            reset_run_memory(Config(device="cpu"))
            self.assertEqual(events, [])
            reset_run_memory(Config())
        self.assertEqual(events, ["collect", "release", "reset"])
        self.assertEqual(allocator["peak"], 1)

    def test_gate_telemetry_measures_within_image_variation(self):
        """Distinguish spatially varying gates from image-common gates with the same average."""
        local = torch.randn(2, 4, 7, requires_grad=True)
        areas = torch.tensor([0.1, 0.2, 0.3, 0.4])
        gates = torch.tensor([[0.5, 0.7, 0.9, 1.0], [0.86, 0.86, 0.86, 0.86]])
        with patch(
            "clvision.losses.route_gates", return_value=(gates, torch.zeros_like(gates), None)
        ):
            loss, _, metrics = consistency_loss(local, areas, torch.arange(2), Config())
        self.assertAlmostEqual(metrics["joint_fraction"].item(), 0.86, places=6)
        self.assertAlmostEqual(metrics["gate_within_image_std"].item(), 0.0264**0.5 / 2, places=6)
        self.assertAlmostEqual(metrics["joint_saturated_fraction"].item(), 0.2, places=6)
        self.assertFalse(any(value.requires_grad for value in metrics.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_loss_diagnostics_remain_detached_until_batched_host_transfer(self):
        """The loss must not extract Python scalars before returning its differentiable result."""
        local = torch.randn(4, 4, 7, requires_grad=True)
        areas = torch.full((4,), 0.25)
        targets = torch.arange(4)
        with patch.object(
            torch.Tensor, "__float__", side_effect=AssertionError("early scalar sync")
        ):
            loss, _, metrics = consistency_loss(local, areas, targets, Config())
        self.assertTrue(
            all(
                isinstance(value, torch.Tensor) and not value.requires_grad
                for value in metrics.values()
            )
        )
        recorded = diagnostic_scalars(loss, metrics)
        self.assertTrue(all(type(value) is float for value in recorded.values()))
        for key, value in metrics.items():
            self.assertEqual(recorded[key], value.item())
        self.assertEqual(recorded["loss"], loss.item())
        loss.backward()
        self.assertTrue(torch.isfinite(local.grad).all())
        json.dumps(recorded)

    def test_data_wait_excludes_time_spent_in_batch_work(self):
        """A slow worker startup/fetch counts as data wait; arbitrary consumer work does not."""
        clock = {"seconds": 0.0}

        class Loader:
            """Charge iterator construction separately from fetching and exhausting its batches."""

            def __iter__(self):
                clock["seconds"] += 5

                def batches():
                    """Yield two batches with distinct fetch delays and a teardown delay."""
                    clock["seconds"] += 2
                    yield 1
                    clock["seconds"] += 3
                    yield 2
                    clock["seconds"] += 4

                return batches()

        timings = {"data_wait_seconds": 7}
        seen = []
        with patch("clvision.runtime.time.monotonic", side_effect=lambda: clock["seconds"]):
            for batch in timed_batches(Loader(), timings):
                seen.append(batch)
                clock["seconds"] += 100
        self.assertEqual(seen, [1, 2])
        self.assertEqual(timings["data_wait_seconds"], 21)

    def test_status_identifies_actual_process_settings_and_phase(self):
        """An observer can tell a batch-128 old job from a batch-1024 job and its current phase."""
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                batch_size=1024, eval_batch_size=2048, precision="bf16", max_vram_gib=28
            )
            with (
                patch("clvision.runtime.torch.cuda.memory_allocated", return_value=2**30),
                patch("clvision.runtime.torch.cuda.memory_reserved", return_value=2 * 2**30),
                patch("clvision.runtime.torch.cuda.max_memory_allocated", return_value=2**30),
                patch("clvision.runtime.torch.cuda.max_memory_reserved", return_value=3 * 2**30),
            ):
                runtime_status(directory, config, "validation", epoch=3)
            status = read_json(Path(directory) / "runtime_status.json")
            self.assertEqual(status["pid"], os.getpid())
            self.assertEqual((status["train_batch_size"], status["eval_batch_size"]), (1024, 2048))
            self.assertEqual((status["stage"], status["epoch"]), ("validation", 3))
            self.assertEqual(status["vram"]["reserved_gib"], 2)
            self.assertEqual(status["vram"]["peak_allocated_gib"], 1)
            self.assertEqual(status["vram"]["peak_reserved_gib"], 3)
            self.assertEqual(status["vram_budget_gib"], 28)
