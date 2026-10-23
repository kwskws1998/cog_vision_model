"""Verify throughput excludes cold startup and warmup while preserving all preflight updates."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from clvision.config import recipe
from clvision.profile import profile_method


class ProfileTimingTests(unittest.TestCase):
    def test_cold_startup_and_warmup_do_not_become_per_image_costs(self):
        """Use deterministic fake delays to separate first-batch startup from warmed throughput."""
        clock = {"seconds": 0.0}
        updates, evaluations = [], []
        model = torch.nn.Linear(1, 1)
        model.shared_state_hash = lambda: "initial"
        config = recipe(
            "cifar100",
            device="cpu",
            precision="fp32",
            debug=True,
            workers=0,
            batch_size=2,
            eval_batch_size=4,
        )

        def loader(manifest, config, split, epoch=0):
            """Charge construction once, a slow first batch, and unit cost for subsequent batches."""
            clock["seconds"] += 1

            def batches():
                """Model real iterator fetch costs without CUDA or operating-system scheduling."""
                for index in range(20):
                    clock["seconds"] += (50 if split == "train" else 30) if index == 0 else 1
                    count = config.batch_size if split == "train" else config.eval_batch_size
                    yield torch.ones(count, 1), torch.zeros(count, dtype=torch.long)

            return batches()

        def optimize(model, optimizer, scaler, images, targets, config):
            """Make the first five accepted updates deliberately slower than steady updates."""
            clock["seconds"] += 10 if len(updates) < 5 else 2
            updates.append(len(images))
            with torch.no_grad():
                model.weight.add_(0.01)
            return {"loss": 1.0}

        def evaluate(model, batches, config):
            """Make the first inference cold, then include streaming batch fetch and forward costs."""
            if not evaluations:
                clock["seconds"] += 100
            count = 0
            for images, _ in batches:
                count += len(images)
                clock["seconds"] += 2
            evaluations.append(count)
            return {"n": count}, None, None

        replacements = {
            "time.monotonic": lambda: clock["seconds"],
            "make_model": lambda config: model,
            "make_loader": loader,
            "optimize_batch": optimize,
            "evaluate_loader": evaluate,
            "vram_stats": lambda config: {"peak_reserved_gib": 0},
            "torch.cuda.empty_cache": lambda: None,
            "torch.cuda.reset_peak_memory_stats": lambda: None,
            "torch.cuda.synchronize": lambda: None,
        }
        activity = {}
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch("clvision.profile." + name, replacement))
            result = profile_method({}, config, 10, activity)
        self.assertEqual(len(updates), 10)
        self.assertEqual(activity["accepted_optimizer_steps"], 10)
        self.assertTrue(result["backbone_updated"])
        self.assertEqual(result["training_warmup_steps"], 5)
        self.assertEqual(result["training_measured_steps"], 5)
        self.assertEqual(result["training_measured_images"], 10)
        self.assertEqual(result["training_loader_startup_seconds"], 51)
        self.assertAlmostEqual(result["training_images_per_second"], 10 / 15)
        self.assertEqual(evaluations, [4, 40])
        self.assertEqual(result["evaluation_loader_startup_seconds"], 31)
        self.assertEqual(result["evaluation_measured_images"], 40)
        self.assertAlmostEqual(result["evaluation_images_per_second"], 40 / 29)


if __name__ == "__main__":
    unittest.main()
