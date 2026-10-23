"""User-configured recipes, matched tuning budgets, and immutable study resume contracts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clvision.config import MAIN_METHODS, MECHANISM_METHODS, recipe
from clvision.data import make_fixture
from clvision.settings import StudySettings, resolve_settings
from clvision.workflow import (
    base_recipe,
    candidate_configs,
    choose_candidates,
    freeze_plan,
    run_study,
    selected_config,
)

from cue_routing.io import atomic_json

ROOT = Path(__file__).resolve().parents[2]


class ClassificationSettingsTests(unittest.TestCase):
    """Check externally meaningful configuration behavior without launching GPU training."""

    def setUp(self):
        """Isolate saved plans, fixture manifests, and mocked run artifacts."""
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.manifests = {
            name: make_fixture(self.folder / "data", name)
            for name in ("cifar100", "tiny_imagenet", "dominoes")
        }

    def tearDown(self):
        """Remove only this test's temporary data and run artifacts."""
        self.temporary.cleanup()

    def custom_settings(self):
        """Declare nondefault schedules, seed counts, coefficients, and matched tuning budgets."""
        return StudySettings.from_dict(
            {
                "workers": 0,
                "main_seeds": [31, 32],
                "mechanism_seeds": [41],
                "development_seed": 97,
                "training": {"batch_size": 16, "eval_batch_size": 32, "learning_rate": 0.02},
                "natural_schedule": {"epochs": 8, "milestones": [4, 6]},
                "mechanism_schedule": {"epochs": 4, "milestones": [2]},
                "dataset_overrides": {"tiny_imagenet": {"learning_rate": 0.03}},
                "development_candidates": {
                    dataset: {
                        "cue": [0.1, 0.2],
                        "fixed_global_local": [0.2, 0.4],
                        "zipfls_dense2": [0.1, 0.2],
                    }
                    for dataset in ("cifar100", "tiny_imagenet")
                },
            }
        )

    def test_production_epochs_batches_and_smoothing_are_editable(self):
        """Keep numerical validation while allowing explicit changes to the benchmark recipe."""
        config = recipe("cifar100", epochs=7, milestones=(3, 5), batch_size=16, label_smoothing=0.2)
        self.assertEqual((config.epochs, config.batch_size, config.label_smoothing), (7, 16, 0.2))
        for updates in (
            {"epochs": 0},
            {"batch_size": 1},
            {"label_smoothing": 1.0},
            {"learning_rate": float("nan")},
            {"epochs": 7, "milestones": (7,)},
            {"debug_max_batches": 1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                recipe("cifar100", **updates)

    def test_overrides_propagate_with_explicit_dataset_precedence(self):
        """Common inputs reach all datasets while schedules and dataset overrides stay scoped."""
        settings = self.custom_settings()
        configs = {name: base_recipe(name, settings) for name in self.manifests}
        for config in configs.values():
            self.assertEqual(
                (config.batch_size, config.eval_batch_size, config.workers), (16, 32, 0)
            )
        self.assertEqual(configs["cifar100"].learning_rate, 0.02)
        self.assertEqual(configs["tiny_imagenet"].learning_rate, 0.03)
        self.assertEqual(
            (configs["tiny_imagenet"].epochs, configs["tiny_imagenet"].milestones), (8, (4, 6))
        )
        self.assertEqual((configs["dominoes"].epochs, configs["dominoes"].milestones), (4, (2,)))

    def test_seed_lists_and_full_recipe_are_frozen(self):
        """The plan records every effective setting and development trial before scores exist."""
        settings = self.custom_settings()
        plan = freeze_plan(self.folder / "study", self.manifests, ROOT, settings)
        self.assertEqual(plan["main_seeds"], [31, 32])
        self.assertEqual(plan["mechanism_seeds"], [41])
        self.assertEqual(plan["settings"], settings.to_dict())
        self.assertEqual(plan["resolved_recipes"]["dominoes"]["epochs"], 4)
        self.assertEqual(len(plan["development_trials"]["cifar100"]), 6)
        self.assertTrue(
            all(row["config"]["seed"] == 97 for row in plan["development_trials"]["cifar100"])
        )
        self.assertEqual(plan, freeze_plan(self.folder / "study", self.manifests, ROOT, settings))

    def test_arbitrary_equal_candidate_counts_keep_validation_selection(self):
        """Changing the tuning budget preserves matched trials and the validation tiebreak."""
        settings = self.custom_settings()
        candidates = candidate_configs(base_recipe("cifar100", settings), settings)
        self.assertEqual(len(candidates), 6)
        self.assertEqual({config.seed for _, config in candidates}, {97})
        rows = [
            {
                "name": name,
                "config": config.to_dict(),
                "validation": {"accuracy": 0.5, "nll": 2.0 if index % 2 == 0 else 1.0},
            }
            for index, (name, config) in enumerate(candidates)
        ]
        chosen = choose_candidates(rows)
        self.assertEqual(chosen["cue"]["gate_tau"], 0.2)
        self.assertEqual(chosen["fixed_global_local"]["fixed_lambda"], 0.4)
        with self.assertRaises(ValueError):
            choose_candidates(rows[:-1])

    def test_invalid_inputs_are_rejected_before_execution(self):
        """Reject misspellings, empty or duplicate seeds, invalid numbers, and unequal search."""
        invalid = (
            {"main_seeds": []},
            {"main_seeds": [1, 1]},
            {"main_seeds": [True]},
            {"mechanism_seeds": [2**63]},
            {"development_seed": -1},
            {"workers": True},
            {"debug": "false"},
            {"epoch": 2},
            {"training": {"method": "cue"}},
            {"training": {"learning_rate": float("inf")}},
            {"natural_schedule": {"milestones": None}},
            {"dataset_overrides": {"typo": {}}},
            {"development_candidates": {"cifar100": {"cue": [0.2]}}},
            {"development_candidates": {"cifar100": {"cue": [0.1, 0.1, 0.2]}}},
            {"development_candidates": {"cifar100": {"cue": [-0.1, 0.1, 0.2]}}},
            {"debug": True, "training": {"batch_size": 0}},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StudySettings.from_dict(values)

    def test_tuned_coefficients_cannot_be_silently_overwritten(self):
        """Direct tunable overrides fail with an actionable route to explicit candidate grids."""
        for key in ("gate_tau", "fixed_lambda", "zipf_lambda"):
            for values in (
                {"training": {key: 0.8}},
                {"dataset_overrides": {"cifar100": {key: 0.8}}},
                {"dataset_overrides": {"dominoes": {key: 0.8}}},
            ):
                with (
                    self.subTest(values=values),
                    self.assertRaisesRegex(ValueError, "development_candidates"),
                ):
                    StudySettings.from_dict(values)

    def test_singleton_grids_preserve_explicit_fixed_coefficients(self):
        """A matched one-trial grid carries user-selected coefficients into the main runs."""
        settings = StudySettings.from_dict(
            {
                "development_candidates": {
                    "cifar100": {
                        "cue": [0.8],
                        "fixed_global_local": [0.7],
                        "zipfls_dense2": [0.6],
                    }
                }
            }
        )
        base = base_recipe("cifar100", settings)
        candidates = candidate_configs(base, settings)
        self.assertEqual(len(candidates), 3)
        selection = {
            "selection_data": "internal_validation_only_final_epoch",
            "selected": choose_candidates(
                [
                    {
                        "config": config.to_dict(),
                        "validation": {"accuracy": 0.5, "nll": 1.0},
                    }
                    for _, config in candidates
                ]
            ),
        }
        for method, key, expected in (
            ("cue", "gate_tau", 0.8),
            ("fixed_global_local", "fixed_lambda", 0.7),
            ("zipfls_dense2", "zipf_lambda", 0.6),
        ):
            config = selected_config(base, selection, method, seed=12)
            self.assertEqual(getattr(config, key), expected)
            self.assertEqual(config.seed, 12)

    def test_example_round_trip_and_legacy_debug_defaults(self):
        """The checked-in config loads directly and legacy smoke calls remain small."""
        settings = StudySettings.load(ROOT / "classification/configs/experiment.json")
        self.assertEqual(settings.main_seeds, (0, 1))
        self.assertEqual(settings.to_dict(), StudySettings.from_dict(settings.to_dict()).to_dict())
        smoke = resolve_settings({"debug": True, "workers": 0})
        self.assertEqual((smoke.main_seeds, smoke.mechanism_seeds), ((0,), (0,)))
        for dataset in self.manifests:
            config = base_recipe(dataset, smoke)
            self.assertEqual(
                (config.device, config.precision, config.epochs, config.debug_max_batches),
                ("cpu", "fp32", 2, 2),
            )
        explicit = resolve_settings({"debug": True, "main_seeds": [5, 8]})
        self.assertEqual(explicit.main_seeds, (5, 8))

    def test_changed_inputs_cannot_resume_a_frozen_study(self):
        """Seeds, optimizer settings, dataset settings, and candidate order bind the saved plan."""
        settings = self.custom_settings()
        directory = self.folder / "study"
        freeze_plan(directory, self.manifests, ROOT, settings)
        mutations = (
            {"main_seeds": [31]},
            {"mechanism_seeds": [42]},
            {"development_seed": 98},
            {"training": {**settings.training, "learning_rate": 0.04}},
            {"natural_schedule": {"epochs": 9, "milestones": [4, 6]}},
            {"dataset_overrides": {}},
            {
                "development_candidates": {
                    dataset: {**methods, "cue": list(reversed(methods["cue"]))}
                    for dataset, methods in settings.development_candidates.items()
                }
            },
        )
        for update in mutations:
            changed = StudySettings.from_dict({**settings.to_dict(), **update})
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "resume cannot"):
                freeze_plan(directory, self.manifests, ROOT, changed)

    def test_changed_resume_is_rejected_before_cuda_preflight(self):
        """A stale study must fail before profiling can perform any optimizer updates."""
        settings = self.custom_settings()
        manifests = {**self.manifests, "cifar100_c": {"dataset_id": "corruptions"}}
        directory = self.folder / "study"
        freeze_plan(directory, manifests, ROOT, settings)
        changed = StudySettings.from_dict({**settings.to_dict(), "main_seeds": [100]})
        with (
            patch(
                "clvision.workflow.load_manifest",
                side_effect=lambda path, **kwargs: manifests[Path(path).stem],
            ),
            patch("clvision.workflow.profile_dataset") as profile,
            patch("clvision.workflow.train") as train,
            self.assertRaisesRegex(ValueError, "resume cannot"),
        ):
            run_study(ROOT, self.folder / "data", directory, settings=changed)
        profile.assert_not_called()
        train.assert_not_called()

    def test_preflight_and_training_use_the_same_resolved_settings(self):
        """Profile and all training phases honor nondefault schedules, batches, LR, and seed scope."""
        settings = self.custom_settings()
        manifests = {**self.manifests, "cifar100_c": {"dataset_id": "corruptions"}}
        saved, profiled, training = {}, {}, []

        def fake_profile(root, manifest, config, path):
            profiled[config.dataset] = config
            return {}

        def fake_train(
            config, manifest, directory, root, phase="main", selection_id="", resume=False
        ):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            saved[str(directory)] = phase
            training.append((phase, config))
            atomic_json(directory / "completion.json", {"config": config.to_dict()})

        def fake_verify(directory, *args, **kwargs):
            return (
                {"phase": saved[str(directory)]},
                {},
                {"final_validation": {"accuracy": 0.5, "nll": 1.0}},
            )

        with (
            patch(
                "clvision.workflow.load_manifest",
                side_effect=lambda path, **kwargs: manifests[Path(path).stem],
            ),
            patch("clvision.workflow.profile_dataset", side_effect=fake_profile),
            patch("clvision.workflow.verify_profile"),
            patch("clvision.workflow.train", side_effect=fake_train),
            patch("clvision.workflow.verify_run", side_effect=fake_verify),
            patch("clvision.workflow.evaluate_run"),
            patch("clvision.workflow.evaluate_corruptions"),
            patch("clvision.workflow.report", return_value={"passed": True}),
        ):
            result = run_study(ROOT, self.folder / "data", self.folder / "study", settings=settings)
        self.assertTrue(result["passed"])
        self.assertEqual(len(profiled), 3)
        self.assertEqual(len(training), 12 + 2 * 2 * len(MAIN_METHODS) + len(MECHANISM_METHODS))
        for phase, config in training:
            base = profiled[config.dataset]
            for key in (
                "epochs",
                "milestones",
                "batch_size",
                "eval_batch_size",
                "learning_rate",
                "workers",
                "precision",
            ):
                self.assertEqual(getattr(config, key), getattr(base, key))
            expected_seeds = {"development": {97}, "main": {31, 32}, "mechanism": {41}}
            self.assertIn(config.seed, expected_seeds[phase])
