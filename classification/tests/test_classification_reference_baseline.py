"""Independent author-code oracles and checkpoint/prediction tests for reference CE reproduction."""

import ast
import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torchvision
from clvision.baseline_reference import (
    REFERENCE,
    BaselineConfig,
    build_model,
    checked_reference,
    image_transform,
    load_data,
    make_loaders,
)
from clvision.baseline_training import phase, train_seed
from clvision.display import DisplayOptions, display_options
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from cue_routing.io import file_hash, read_json


def author_function(filename, name, globals_dict):
    """Execute only the inspected reference function, avoiding unrelated author's service imports."""
    path = REFERENCE / f"pytorch_zipfls__{filename}.txt"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), globals_dict)
    return globals_dict[name]


class ReferenceBaselineTests(unittest.TestCase):
    """Validate the full protocol independently of its own printed recipe."""

    def setUp(self):
        """Isolate checkpoints, random state, console output and CPU execution settings."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(2)
        self.enterContext(torch.random.fork_rng(devices=[]))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(display_options(DisplayOptions(progress="off")))

    def test_defaults_and_pinned_hashes(self):
        """The runnable preset must select exactly one CE baseline and the audited upstream files."""
        config = BaselineConfig().validate()
        self.assertEqual(config.seeds, [0])
        self.assertEqual(config.preview()["training_examples"], 50000)
        self.assertEqual(config.preview()["evaluation_examples"], 10000)
        self.assertEqual(config.preview()["lr_drop_first_used_epochs"], [102, 152])
        self.assertGreaterEqual(len(checked_reference()["files"]), 9)

    def test_lr_matches_author_loop_including_end_of_epoch_shift(self):
        """Compare every rate actually used in 200 steps of the author's epoch-level LR loop."""
        adjust = author_function("utils.py", "adjust_learning_rate", {})
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
        config = BaselineConfig()
        for zero_based_epoch in range(200):
            self.assertAlmostEqual(
                config.rate(zero_based_epoch + 1), optimizer.param_groups[0]["lr"]
            )
            adjust("CIFAR100", optimizer, zero_based_epoch, 0.1, 200)
        paper = replace(config, schedule="paper")
        self.assertAlmostEqual(paper.rate(100), 0.1)
        self.assertAlmostEqual(paper.rate(101), 0.01)
        self.assertAlmostEqual(paper.rate(151), 0.001)

    def test_transforms_and_loader_semantics_match_author(self):
        """Compare augmented pixels under identical RNGs against the actual upstream loader function."""
        seen = []

        def fake_dataset(*args, **kwargs):
            """Record author transforms without downloading CIFAR data."""
            result = SimpleNamespace(transform=kwargs["transform"], train=kwargs["train"])
            seen.append(result)
            return result

        load = author_function(
            "datasets.py",
            "load_dataset",
            {
                "transforms": transforms,
                "torchvision": torchvision,
                "DataLoader": lambda data, **kwargs: SimpleNamespace(dataset=data, **kwargs),
            },
        )
        with patch.object(torchvision.datasets, "CIFAR100", side_effect=fake_dataset):
            train, test = load("CIFAR100", "unused", batchsize=128)
        self.assertTrue(train.shuffle)
        self.assertFalse(test.shuffle)
        self.assertTrue(seen[0].train)
        self.assertFalse(seen[1].train)
        image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
        for training, data in ((True, train), (False, test)):
            torch.manual_seed(103)
            expected = data.dataset.transform(image)
            torch.manual_seed(103)
            actual = image_transform(training)(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_dataset_paths_counts_and_last_short_batch(self):
        """Use a metadata-only CIFAR fixture to catch silent 45k subsets or train/test swaps."""
        real_class = torchvision.datasets.CIFAR100
        calls = []

        class MetadataCifar(real_class):
            """Provide official dimensions and labels without allocating 50k image arrays."""

            def __init__(self, root, train, transform, download):
                self.n = 50000 if train else 10000
                self.data = SimpleNamespace(shape=(self.n, 32, 32, 3))
                self.targets = [i % 100 for i in range(self.n)]
                calls.append((Path(root), train))

            def __len__(self):
                return self.n

        raw = self.root / "raw/cifar100/checked" / real_class.base_folder
        raw.mkdir(parents=True)
        for name in ("train", "test", "meta"):
            (raw / name).write_text(name)
        with patch.object(torchvision.datasets, "CIFAR100", MetadataCifar):
            train, test, identity = load_data(self.root)
        self.assertEqual((len(train), len(test)), (50000, 10000))
        self.assertEqual([flag for _, flag in calls], [True, False])
        self.assertEqual(calls[0][0], raw.parent.resolve())
        self.assertNotEqual(identity["sha256"]["train"], identity["sha256"]["test"])
        a, b = make_loaders(train, test, BaselineConfig(device="cpu", workers=0))
        self.assertEqual((len(a), len(b)), (391, 79))
        self.assertFalse(a.drop_last)

    def test_ce_step_and_auxiliary_heads(self):
        """Match an independent CE/SGD update and ensure the unused heads receive no gradients."""
        torch.manual_seed(101)
        actual = build_model()
        expected = copy.deepcopy(actual)
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([3, 7])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        first = torch.optim.SGD(actual.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        second = torch.optim.SGD(expected.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        phase(actual, loader, "cpu", first)
        expected.train()
        torch.nn.functional.cross_entropy(expected(images)[0][0], labels).backward()
        second.step()
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
        self.assertIsNone(actual.linear2.weight.grad)
        self.assertIsNone(actual.linear3.weight.grad)
        before = copy.deepcopy(actual.state_dict())
        metrics, outputs = phase(actual, loader, "cpu", save_predictions=True)
        self.assertEqual(
            metrics["correct"], int(outputs["logits"].argmax(1).eq(outputs["labels"]).sum())
        )
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_resume_and_saved_final_test_predictions(self):
        """Interrupted training must yield identical final weights and independently countable scores."""
        config = BaselineConfig(
            epochs=2, batch_size=2, eval_batch_size=2, workers=0, device="cpu", milestones=[1]
        )
        torch.manual_seed(4)
        train = TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2]))
        test = TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([8, 9, 10]))
        identity = {"kind": "synthetic_smoke_not_benchmark", "train_n": 3, "test_n": 3}
        a, b = self.root / "full", self.root / "resume"
        train_seed(config, 0, train, test, identity, a)
        train_seed(config, 0, train, test, identity, b, stop_after_epoch=1)
        summary = train_seed(config, 0, train, test, identity, b, resume=True)
        left = torch.load(a / "last.pt", weights_only=True)
        right = torch.load(b / "last.pt", weights_only=True)
        for key, value in left["model"].items():
            torch.testing.assert_close(value, right["model"][key], rtol=0, atol=0)
        for key, item in left["optimizer"]["state"].items():
            torch.testing.assert_close(
                item["momentum_buffer"],
                right["optimizer"]["state"][key]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
        saved = torch.load(b / "test_predictions.pt", weights_only=True)
        self.assertEqual(saved["split"], "synthetic_test")
        self.assertEqual(summary["evaluation_split"], "synthetic_test")
        self.assertEqual(summary["final_test"]["n"], 3)
        self.assertTrue(torch.equal(saved["labels"], test.tensors[1]))
        self.assertEqual(summary["predictions_sha256"], file_hash(b / "test_predictions.pt"))
        self.assertEqual(
            summary["final_test"]["correct"],
            int(saved["logits"].argmax(1).eq(saved["labels"]).sum()),
        )
        self.assertEqual(read_json(b / "status.json")["stage"], "complete")
        with self.assertRaises(FileExistsError):
            train_seed(config, 0, train, test, identity, b)
        with self.assertRaisesRegex(ValueError, "contract differs"):
            train_seed(replace(config, learning_rate=0.2), 0, train, test, identity, b, resume=True)

    def test_reject_invalid_inputs(self):
        """Fail fast on malformed sweep inputs rather than launching an unintended long run."""
        for values in (
            {"seeds": []},
            {"seeds": [0, 0]},
            {"batch_size": 0},
            {"milestones": [150, 100]},
            {"learning_rate": float("nan")},
            {"momentum": 1.0},
            {"schedule": "unknown"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BaselineConfig(**values).validate()
