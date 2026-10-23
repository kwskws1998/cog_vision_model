"""Office-Home 65-output training, exact resume, target seals, and complete matrix checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from ohvision.cli import matrix, report_matrix
from ohvision.config import CLASSES, DOMAINS, METHODS, ExperimentConfig
from ohvision.data import evaluation_loader, make_fixture
from ohvision.engine import evaluate_target, make_model, train
from ohvision.fingerprint import source_hash

from cue_routing.io import (
    atomic_json,
    atomic_torch,
    canonical_hash,
    environment,
    file_hash,
    read_json,
    seed_everything,
    state_hash,
)
from cue_routing.metrics import classification_metrics
from cue_routing.models import CueReadout
from cue_routing.protocol import record_target_access, seal_selection

ROOT = Path(__file__).resolve().parents[2]


class TinyOfficeClassifier(torch.nn.Module):
    """Use a small encoder with the exact 65-output cue/RSC interfaces for protocol tests."""

    def __init__(self):
        """Keep the real readout operator while shrinking only the artificial-test encoder."""
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        self.readout = CueReadout(8, len(CLASSES))

    @property
    def classifier(self):
        """Expose the shared classifier to the unchanged RSC operator."""
        return self.readout.classifier

    def forward_features(self, images):
        """Return BCHW feature maps before the common readout."""
        return self.encoder(images)

    def forward(self, images):
        """Use identical global pooling at evaluation for every training method."""
        return self.readout(self.forward_features(images))


def completed_matrix(directory, seeds=(0,)):
    """Build sealed synthetic artifacts to test reporting, never to claim real-data accuracy."""
    directory = Path(directory)
    config = ExperimentConfig(device="cpu", precision="fp32")
    specifications = [
        {"method": method, "target": domain, "seed": seed}
        for seed in sorted(seeds)
        for domain in DOMAINS
        for method in METHODS
    ]
    plan = {
        "schema": 2,
        "base_config": config.to_dict(),
        "runs": specifications,
        "dataset_id": "artificial-report-fixture",
        "source_sha256": source_hash(ROOT),
    }
    atomic_json(directory / "matrix_plan.json", plan)
    for specification in specifications:
        run_dir = (
            directory
            / f"{specification['target']}_seed{specification['seed']}_{specification['method']}"
        )
        labels = torch.tensor([0, 32, 64])
        paths = [
            f"{specification['target']}/{CLASSES[label]}/fixture.jpg" for label in labels.tolist()
        ]
        contract = {
            "config": {**config.to_dict(), **specification},
            "dataset_kind": "officehome",
            "class_order": list(CLASSES),
            "domains": list(DOMAINS),
            "dataset_id": plan["dataset_id"],
            "source_sha256": plan["source_sha256"],
            "runtime": environment(),
            "initial_model_sha256": f"same-{specification['seed']}",
            "target_n": len(paths),
            "target_partition_sha256": canonical_hash(
                [[path, label] for path, label in zip(paths, labels.tolist())]
            ),
        }
        atomic_json(run_dir / "contract.json", contract)
        atomic_torch(run_dir / "best.pt", {"artificial_unit_test": torch.zeros(1)})
        selection = seal_selection(run_dir, contract, {"step": config.steps}, config.steps)
        logits = torch.nn.functional.one_hot(labels, len(CLASSES)).float() * 4
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
                "contract_hash": canonical_hash(contract),
                "selection": selection,
                "prediction_sha256": file_hash(run_dir / "target_predictions.pt"),
                "metrics": classification_metrics(logits, labels),
            },
        )
        record_target_access(run_dir, selection)
    return config, plan


class OfficeHomeTrainingTests(unittest.TestCase):
    """Verify actual 65-class optimizer paths and resume without using target labels."""

    @classmethod
    def setUpClass(cls):
        """Prepare a complete artificial class vocabulary once for the protocol checks."""
        cls.temporary = tempfile.TemporaryDirectory()
        cls.manifest = make_fixture(Path(cls.temporary.name) / "images")

    @classmethod
    def tearDownClass(cls):
        """Remove artificial dataset files after testing."""
        cls.temporary.cleanup()

    def config(self, method="cue"):
        """Return a clearly marked, short CPU recipe for numerical tests."""
        return ExperimentConfig(
            method=method,
            steps=4,
            eval_every=2,
            image_size=64,
            batch_per_domain=1,
            eval_batch_size=65,
            workers=0,
            device="cpu",
            precision="fp32",
            pretrained=False,
            debug=True,
        )

    def test_real_resnet_has_65_outputs_and_expected_parameter_count(self):
        seed_everything(0)
        model = make_model(self.config(), ROOT)
        self.assertEqual(model.classifier.out_features, 65)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 11209857)
        model.eval()
        with torch.no_grad():
            self.assertEqual(model(torch.zeros(2, 3, 64, 64)).shape, (2, 65))

    def test_exact_resume_all_methods_and_final_target_partition(self):
        for method in METHODS:
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as temporary,
                patch(
                    "ohvision.engine.make_model",
                    side_effect=lambda *args, **kwargs: TinyOfficeClassifier(),
                ),
            ):
                root = Path(temporary)
                config = self.config(method)
                train(config, self.manifest, root / "full", ROOT)
                train(config, self.manifest, root / "resumed", ROOT, stop_after=2)
                train(config, self.manifest, root / "resumed", ROOT, resume=True)
                full = torch.load(root / "full/last.pt", map_location="cpu", weights_only=True)
                resumed = torch.load(
                    root / "resumed/last.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(state_hash(full["model"]), state_hash(resumed["model"]))
                for index, state in full["optimizer"]["state"].items():
                    for key, value in state.items():
                        self.assertTrue(
                            torch.equal(value, resumed["optimizer"]["state"][index][key])
                        )
                result = evaluate_target(self.manifest, root / "resumed", ROOT)
                self.assertFalse(result["benchmark_evidence"])
                self.assertEqual(result["metrics"]["n"], 130)
                predictions = torch.load(
                    root / "resumed/target_predictions.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(predictions["logits"].shape, (130, 65))
                with self.assertRaisesRegex(ValueError, "after target"):
                    train(config, self.manifest, root / "resumed", ROOT, resume=True)

    def test_source_selection_never_constructs_target_or_test_loader(self):
        config = self.config()

        def guarded(manifest, domain, split, recipe):
            """Fail immediately if source-only selection attempts any held-out access."""
            self.assertNotEqual(domain, config.target)
            self.assertEqual(split, "crossval")
            return evaluation_loader(manifest, domain, split, recipe)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ohvision.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyOfficeClassifier(),
            ),
            patch("ohvision.engine.evaluation_loader", side_effect=guarded),
        ):
            train(config, self.manifest, temporary, ROOT)

    def test_resume_rejects_changed_recipe(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ohvision.engine.make_model",
                side_effect=lambda *args, **kwargs: TinyOfficeClassifier(),
            ),
        ):
            config = self.config()
            train(config, self.manifest, temporary, ROOT, stop_after=2)
            changed = ExperimentConfig.from_dict({**config.to_dict(), "learning_rate": 0.001})
            with self.assertRaisesRegex(ValueError, "Resume requires identical"):
                train(changed, self.manifest, temporary, ROOT, resume=True)


class OfficeHomeMatrixTests(unittest.TestCase):
    """Check the full 36-run schedule, paired artifacts, reporting, and repeated invocation."""

    def test_all_36_training_runs_precede_every_target_evaluation(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ohvision.cli.load_manifest",
                return_value={
                    "kind": "officehome",
                    "variant": "original_resolution",
                    "dataset_id": "artificial-matrix",
                },
            ),
            patch("ohvision.cli.validate_matrix_runs", return_value=set()),
            patch("ohvision.cli.train", side_effect=lambda *args, **kwargs: events.append("train")),
            patch(
                "ohvision.cli.evaluate_target",
                side_effect=lambda *args, **kwargs: events.append("target"),
            ),
            patch("ohvision.cli.report_matrix", return_value={"schedule_verified": True}),
        ):
            matrix(
                ExperimentConfig(device="cpu", precision="fp32"),
                "unused",
                temporary,
                ROOT,
                [0, 1, 2],
                evaluate=True,
            )
            plan = read_json(Path(temporary) / "matrix_plan.json")
        self.assertEqual(len(plan["runs"]), 36)
        self.assertEqual(events, ["train"] * 36 + ["target"] * 36)

    def test_completed_matrix_is_reused_and_report_has_officehome_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, plan = completed_matrix(temporary)
            with (
                patch(
                    "ohvision.cli.load_manifest",
                    return_value={
                        "kind": "officehome",
                        "variant": "original_resolution",
                        "dataset_id": plan["dataset_id"],
                    },
                ),
                patch("ohvision.cli.train") as training,
                patch("ohvision.cli.evaluate_target") as evaluation,
            ):
                result = matrix(config, "unused", temporary, ROOT, [0], evaluate=True)
            training.assert_not_called()
            evaluation.assert_not_called()
            self.assertEqual(result["runs"], 12)
            report = (Path(temporary) / "comparison.md").read_text()
            self.assertIn("Clipart | Product | Real World", report)
            self.assertNotIn("Sketch", report)

    def test_report_rejects_unmatched_initial_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed_matrix(temporary)
            directory = Path(temporary) / "Art_seed0_cue"
            contract = read_json(directory / "contract.json")
            contract["initial_model_sha256"] = "unmatched"
            atomic_json(directory / "contract.json", contract)
            seal_selection(directory, contract, {"step": 5000}, 5000)
            with self.assertRaises(ValueError):
                report_matrix(temporary)

    def test_report_rejects_reordered_target_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed_matrix(temporary)
            directory = Path(temporary) / "Art_seed0_cue"
            predictions = torch.load(
                directory / "target_predictions.pt", map_location="cpu", weights_only=True
            )
            predictions["paths"].reverse()
            atomic_torch(directory / "target_predictions.pt", predictions)
            metrics = read_json(directory / "target_metrics.json")
            metrics["prediction_sha256"] = file_hash(directory / "target_predictions.pt")
            atomic_json(directory / "target_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "held-out target partition"):
                report_matrix(temporary)
