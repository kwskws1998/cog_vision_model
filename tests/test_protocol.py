"""Data, source-only selection, tamper checks, metrics, and exact-resume tests."""

import tempfile
import types
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from cue_routing.cli import matrix, report_matrix
from cue_routing.config import DOMAINS, ExperimentConfig
from cue_routing.data import (
    DASSL_PACS_URL,
    BalancedStepSampler,
    download_pacs,
    extract_archive,
    load_manifest,
    make_fixture,
    read_official_splits,
    training_loader,
)
from cue_routing.engine import evaluate_target, train
from cue_routing.io import (
    atomic_json,
    atomic_torch,
    canonical_hash,
    file_hash,
    read_json,
    state_hash,
)
from cue_routing.metrics import aggregate_domains, classification_metrics
from cue_routing.models import CueReadout
from cue_routing.protocol import is_better, seal_selection, verify_selection

ROOT = Path(__file__).resolve().parents[1]


class TinyClassifier(nn.Module):
    """Use a small trainable encoder for checkpoint/protocol tests; smoke uses ResNet-18."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.readout = CueReadout(8, 7)

    @property
    def classifier(self):
        return self.readout.classifier

    def forward_features(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.readout(self.forward_features(images))


class DataTests(unittest.TestCase):
    """Protect partition identity and deterministic balanced mini-batches."""

    def test_official_partition_counts_and_classes(self):
        splits, hashes = read_official_splits(ROOT / "references" / "pacs_splits")
        self.assertEqual(len(hashes), 12)
        self.assertEqual(sum(len(item["test"]) for item in splits.values()), 9991)
        self.assertEqual(len(splits["sketch"]["test"]), 3929)

    def test_sampler_exact_resume_and_balance(self):
        domains = {"a": [0, 1, 2], "b": [3, 4], "c": [5, 6, 7, 8]}
        full = list(BalancedStepSampler(domains, 3, 9, 0, 6))
        resumed = list(BalancedStepSampler(domains, 3, 9, 2, 6))
        self.assertEqual(full[2:], resumed)
        for batch in full:
            for index, values in enumerate(domains.values()):
                self.assertTrue(
                    all(sample[0] in values for sample in batch[index * 3 : (index + 1) * 3])
                )
        self.assertNotEqual(full, list(BalancedStepSampler(domains, 3, 10, 0, 6)))

    def test_stateless_augmentation_and_source_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_fixture(temporary, ROOT / "references" / "pacs_splits")
            manifest = load_manifest(path, ROOT / "references" / "pacs_splits")
            config = ExperimentConfig(
                steps=2,
                image_size=64,
                batch_per_domain=2,
                device="cpu",
                precision="fp32",
                pretrained=False,
                workers=0,
                debug=True,
            )
            loader = training_loader(manifest, config)
            self.assertNotIn(config.target, loader.dataset.domain_indices)
            before = torch.get_rng_state().clone()
            first = loader.dataset[(0, 921)][0]
            second = loader.dataset[(0, 921)][0]
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
            batches = list(loader)
            repeated = list(training_loader(manifest, config))
            for left, right in zip(batches, repeated):
                torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
                torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)
            record = next(iter(manifest["images"]))
            with (Path(temporary) / record).open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                load_manifest(path, ROOT / "references" / "pacs_splits")

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_archive(archive, Path(temporary) / "dataset")
            self.assertFalse((Path(temporary) / "escape.txt").exists())

    def test_nested_zip_image_root_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            manifest = read_json(make_fixture(source, ROOT / "references" / "pacs_splits"))
            archive = temporary / "pacs.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for name in manifest["images"]:
                    handle.write(source / name, "pacs/images/" + name)
            destination = temporary / "imported"
            result = extract_archive(archive, destination)
            self.assertEqual(result["archive_sha256"], file_hash(archive))
            self.assertTrue(all((destination / name).is_file() for name in manifest["images"]))
            with self.assertRaises(FileExistsError):
                extract_archive(archive, destination)

    def test_download_fallback_and_actual_source_provenance(self):
        calls = []

        def downloader(url, output, **kwargs):
            """Simulate an unavailable primary link followed by a valid fallback ZIP."""
            calls.append((url, kwargs))
            if len(calls) == 1:
                return None
            with zipfile.ZipFile(output, "w") as handle:
                handle.writestr("download-test.txt", "test only")
            return output

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict("sys.modules", {"gdown": types.SimpleNamespace(download=downloader)}),
        ):
            destination = Path(temporary) / "PACS.zip"
            result = download_pacs(destination)
            self.assertEqual(result["source_url"], DASSL_PACS_URL)
            self.assertEqual(len(result["earlier_download_failures"]), 1)
            self.assertTrue(destination.exists())
            self.assertTrue(all(not kwargs["use_cookies"] for _, kwargs in calls))

    def test_worker_count_does_not_change_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = make_fixture(temporary, ROOT / "references" / "pacs_splits")
            manifest = load_manifest(path, ROOT / "references" / "pacs_splits")
            config = ExperimentConfig(
                steps=2,
                image_size=64,
                batch_per_domain=1,
                device="cpu",
                precision="fp32",
                pretrained=False,
                workers=0,
                debug=True,
            )
            serial = list(training_loader(manifest, config))
            parallel = list(training_loader(manifest, replace(config, workers=2)))
            for left, right in zip(serial, parallel):
                torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
                torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)


class ProtocolTests(unittest.TestCase):
    """Use real optimizer/checkpoint paths to verify training versus final-test roles."""

    def config(self, method="cue"):
        return ExperimentConfig(
            method=method,
            steps=4,
            eval_every=2,
            image_size=64,
            batch_per_domain=2,
            eval_batch_size=7,
            device="cpu",
            precision="fp32",
            workers=0,
            pretrained=False,
            debug=True,
        )

    def test_selection_tie_rules(self):
        incumbent = {"step": 4, "macro_accuracy": 0.5, "macro_nll": 1.0}
        self.assertTrue(is_better({**incumbent, "macro_accuracy": 0.6}, incumbent))
        self.assertTrue(is_better({**incumbent, "macro_nll": 0.9}, incumbent))
        self.assertTrue(is_better({**incumbent, "step": 2}, incumbent))
        self.assertFalse(is_better({**incumbent, "step": 6}, incumbent))

    def test_exact_resume_all_three_methods_and_target_seal(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyClassifier(),
            ),
        ):
            temporary = Path(temporary)
            manifest = make_fixture(temporary / "images", ROOT / "references" / "pacs_splits")
            for method in ("erm", "rsc", "cue"):
                config = self.config(method)
                full, resumed = temporary / (method + "full"), temporary / (method + "resumed")
                train(config, manifest, full, ROOT)
                train(config, manifest, resumed, ROOT, stop_after=2)
                self.assertFalse((resumed / "selection.json").exists())
                with self.assertRaises(FileNotFoundError):
                    evaluate_target(manifest, resumed, ROOT)
                train(config, manifest, resumed, ROOT, resume=True)
                left = torch.load(full / "last.pt", weights_only=True)
                right = torch.load(resumed / "last.pt", weights_only=True)
                self.assertEqual(state_hash(left["model"]), state_hash(right["model"]))
                self.assertEqual(left["history"], right["history"])
                self.assertEqual(left["best"], right["best"])
                for left_state, right_state in zip(
                    left["optimizer"]["state"].values(), right["optimizer"]["state"].values()
                ):
                    for key in left_state:
                        torch.testing.assert_close(
                            left_state[key], right_state[key], atol=0, rtol=0
                        )
                self.assertFalse((resumed / "target_access.json").exists())
                result = evaluate_target(manifest, resumed, ROOT)
                self.assertEqual(result["metrics"]["n"], 21)
                self.assertFalse(result["benchmark_evidence"])
                with self.assertRaises(ValueError):
                    train(config, manifest, resumed, ROOT, resume=True)
                checkpoint = resumed / "best.pt"
                with checkpoint.open("ab") as handle:
                    handle.write(b"tamper")
                with self.assertRaises(ValueError):
                    verify_selection(resumed, read_json(resumed / "contract.json"))

    def test_resume_rejects_recipe_change(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyClassifier(),
            ),
        ):
            temporary = Path(temporary)
            manifest = make_fixture(temporary / "images", ROOT / "references" / "pacs_splits")
            train(self.config(), manifest, temporary / "run", ROOT, stop_after=2)
            with self.assertRaises(ValueError):
                train(
                    replace(self.config(), gate_tau=0.2),
                    manifest,
                    temporary / "run",
                    ROOT,
                    resume=True,
                )

    def test_recover_completed_checkpoint_before_final_seal(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyClassifier(),
            ),
        ):
            temporary = Path(temporary)
            manifest = make_fixture(temporary / "images", ROOT / "references" / "pacs_splits")
            run_dir = temporary / "run"
            train(self.config(), manifest, run_dir, ROOT)
            selected = torch.load(run_dir / "best.pt", weights_only=True)
            for name in ("selection.json", "best.pt"):
                (run_dir / name).unlink()
            atomic_json(
                run_dir / "summary.json",
                {"status": "interrupted_at_requested_boundary", "completed_steps": 2},
            )
            result = train(self.config(), manifest, run_dir, ROOT, resume=True)
            recovered = torch.load(run_dir / "best.pt", weights_only=True)
            self.assertEqual(state_hash(selected["model"]), state_hash(recovered["model"]))
            self.assertTrue(result["recovered_from_completed_checkpoint"])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_steps"], 4)
            verify_selection(run_dir, read_json(run_dir / "contract.json"))

    def test_matrix_trains_every_arm_before_any_target(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.cli.load_manifest", return_value={"kind": "pacs", "dataset_id": "test"}
            ),
            patch("cue_routing.cli.validate_matrix_runs", return_value=set()),
            patch(
                "cue_routing.cli.train", side_effect=lambda *args, **kwargs: events.append("train")
            ),
            patch(
                "cue_routing.cli.evaluate_target",
                side_effect=lambda *args, **kwargs: events.append("test"),
            ),
            patch("cue_routing.cli.report_matrix", return_value={"verified_schedule": True}),
        ):
            matrix(
                ExperimentConfig(device="cpu", precision="fp32"),
                "unused_manifest.json",
                temporary,
                ROOT,
                [0],
                evaluate=True,
            )
        self.assertEqual(events, ["train"] * 12 + ["test"] * 12)

    def test_report_recomputes_metrics_and_rejects_unmatched_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = ExperimentConfig().to_dict()
            plan = [
                {"method": method, "target": domain, "seed": seed}
                for seed in (0, 1)
                for domain in DOMAINS
                for method in ("erm", "rsc", "cue")
            ]
            atomic_json(
                directory / "matrix_plan.json",
                {
                    "base_config": base,
                    "runs": plan,
                    "dataset_id": "unit-test-fixture",
                    "source_sha256": "unit-test-code",
                },
            )
            for specification in plan:
                run_dir = (
                    directory
                    / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
                )
                paths = [f"{specification['target']}/fixture-{index}.png" for index in range(3)]
                partition = [[path, index] for index, path in enumerate(paths)]
                contract = {
                    "config": {**base, **specification},
                    "dataset_id": "unit-test-fixture",
                    "source_sha256": "unit-test-code",
                    "initial_model_sha256": f"same-initial-{specification['seed']}",
                    "runtime": {"unit_test": True},
                    "target_n": 3,
                    "target_partition_sha256": canonical_hash(partition),
                }
                atomic_json(run_dir / "contract.json", contract)
                atomic_torch(run_dir / "best.pt", {"unit_test_only": torch.tensor([1.0])})
                seal = seal_selection(run_dir, contract, {"step": 5000}, 5000)
                labels = torch.tensor([0, 1, 2])
                predictions = labels.clone()
                mistakes = {"cue": 0, "rsc": 1, "erm": 2}[specification["method"]]
                predictions[:mistakes] = (predictions[:mistakes] + 1) % 7
                logits = torch.nn.functional.one_hot(predictions, 7).float() * 5
                atomic_torch(
                    run_dir / "target_predictions.pt",
                    {
                        "logits": logits,
                        "targets": labels,
                        "paths": paths,
                        "contract_hash": canonical_hash(contract),
                    },
                )
                result = {
                    **specification,
                    "benchmark_evidence": True,
                    "metrics": classification_metrics(logits, labels),
                    "selection": seal,
                    "contract_hash": canonical_hash(contract),
                    "prediction_sha256": file_hash(run_dir / "target_predictions.pt"),
                }
                atomic_json(run_dir / "target_metrics.json", result)
            report = report_matrix(directory)
            self.assertEqual(report["ranking"], ["cue", "rsc", "erm"])
            self.assertAlmostEqual(report["cue_minus_rsc_mean"], 1 / 3)
            self.assertEqual(report["methods"]["cue"]["seed_sd"], 0)
            affected = directory / "sketch_seed0_cue"
            metrics = read_json(affected / "target_metrics.json")
            metrics["metrics"]["accuracy"] = 0.1
            atomic_json(affected / "target_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "metrics do not match"):
                report_matrix(directory)
            metrics["metrics"]["accuracy"] = 1.0
            atomic_json(affected / "target_metrics.json", metrics)
            changed = read_json(affected / "contract.json")
            changed["initial_model_sha256"] = "different-initial-weights"
            atomic_json(affected / "contract.json", changed)
            with self.assertRaises(ValueError):
                report_matrix(directory)

    def test_invalid_configurations_rejected(self):
        for updates in (
            {"learning_rate": float("nan")},
            {"learning_rate": True},
            {"steps": 1.5},
            {"pretrained": "false"},
            {"seed": -1},
            {"seed": 2**63},
            {"joint_floor": 1.1},
            {"grid_size": 3},
            {"precision": "fp16", "device": "cpu"},
        ):
            with self.assertRaises(ValueError):
                replace(self.config(), **updates).validate()
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"misspelling": 1})

    def test_source_validation_never_opens_target_loader(self):
        from cue_routing.data import evaluation_loader

        domains_seen = []

        def guarded_loader(manifest, domain, split, config):
            """Fail immediately if training tries to evaluate a target or test partition."""
            self.assertNotEqual(domain, config.target)
            self.assertEqual(split, "crossval")
            domains_seen.append(domain)
            return evaluation_loader(manifest, domain, split, config)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "cue_routing.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyClassifier(),
            ),
            patch("cue_routing.engine.evaluation_loader", side_effect=guarded_loader),
        ):
            temporary = Path(temporary)
            manifest = make_fixture(temporary / "images", ROOT / "references" / "pacs_splits")
            train(self.config(), manifest, temporary / "run", ROOT)
            self.assertEqual(set(domains_seen), set(DOMAINS) - {"sketch"})


class MetricTests(unittest.TestCase):
    """Check metric units and equal-domain rather than pooled selection weights."""

    def test_uniform_and_perfect_predictions(self):
        uniform = classification_metrics(torch.zeros(7, 7), torch.arange(7))
        self.assertAlmostEqual(uniform["accuracy"], 1 / 7)
        self.assertAlmostEqual(uniform["nll"], float(torch.tensor(7.0).double().log()))
        self.assertAlmostEqual(uniform["brier"], 6 / 7)
        self.assertAlmostEqual(uniform["ece"], 0)
        perfect = classification_metrics(torch.eye(7) * 100, torch.arange(7))
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["balanced_accuracy"], 1)

    def test_equal_domain_aggregation(self):
        result = aggregate_domains(
            {
                "large": {"n": 1000, "accuracy": 1, "nll": 0},
                "small": {"n": 1, "accuracy": 0, "nll": 2},
            }
        )
        self.assertEqual(result["macro_accuracy"], 0.5)
        self.assertEqual(result["macro_nll"], 1)
