"""Verify editable experiment recipes preserve matched settings and validation-only selection."""

import math
import unittest
from dataclasses import replace

from clvision.config import Config
from clvision.experiment_settings import (
    Budget,
    Search,
    cue_grid,
    deadline_time,
    select_candidate,
    validate_stages,
)


class ExperimentSettingsTests(unittest.TestCase):
    """Check real recipe flexibility and reject accidental changes to matched comparison controls."""

    def test_common_hyperparameters_are_user_controlled(self):
        """Accept a non-paper recipe and preserve every common field for both methods."""
        base = Config(
            dataset="tiny_imagenet",
            method="erm",
            activation="postact",
            epochs=17,
            milestones=(9, 14),
            warmup_epochs=3,
            batch_size=256,
            eval_batch_size=512,
            learning_rate=0.23,
            momentum=0.85,
            weight_decay=0.002,
            lr_decay=0.2,
            seed=73,
            precision="bf16",
            eval_precision="fp16",
            gradient_clip=1.2,
            gate_tau=0.32,
            gate_temperature=0.14,
            joint_floor=0.24,
        )
        actual = Search([{}, {"gate_tau": 0.27}]).resolve(base)
        self.assertEqual(actual[0], replace(base, method="cue"))
        self.assertEqual(actual[1], replace(base, method="cue", gate_tau=0.27))
        self.assertEqual(
            Search([{}]).resolve(replace(base, activation="preact"))[0].activation, "preact"
        )

    def test_grid_order_and_partial_overrides(self):
        """Make declared Cartesian order explicit and inherit unspecified CUE values from the base."""
        candidates = cue_grid(joint_floor=[0.5, 0.0], gate_tau=[0.15, 0.05])
        self.assertEqual(
            candidates,
            [
                {"joint_floor": 0.5, "gate_tau": 0.15},
                {"joint_floor": 0.5, "gate_tau": 0.05},
                {"joint_floor": 0.0, "gate_tau": 0.15},
                {"joint_floor": 0.0, "gate_tau": 0.05},
            ],
        )
        resolved = Search(candidates).resolve(Config(method="erm", gate_temperature=0.21))
        self.assertEqual({row.gate_temperature for row in resolved}, {0.21})
        self.assertEqual(cue_grid(), [{}])

    def test_rejects_hidden_common_tuning_and_invalid_candidates(self):
        """Prevent an apparent CUE-only search from changing seeds, optimizer, or data."""
        invalid = (
            [],
            [{"seed": 1}],
            [{"learning_rate": 0.8}],
            [{"dataset": "tiny_imagenet"}],
            [{"gate_tau": math.nan}],
            [{"joint_floor": 1.5}],
            [{"gate_temperature": 0}],
            [{"gate_tau": True}],
        )
        for candidates in invalid:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                Search(candidates).resolve(Config(method="erm"))
        for base in (Config(), Config(method="erm", dataset="dominoes")):
            with self.subTest(base=base), self.assertRaises(ValueError):
                Search([{}]).resolve(base)
        with self.assertRaisesRegex(ValueError, "Duplicate resolved"):
            Search([{}, {"gate_tau": 0.15}]).resolve(Config(method="erm"))
        with self.assertRaises(ValueError):
            cue_grid(gate_tau=[])
        with self.assertRaises(ValueError):
            cue_grid(gate_temperature=0.05)
        with self.assertRaises(ValueError):
            cue_grid(learning_rate=[0.1])

    def test_budget_optional_and_timezone_explicit(self):
        """Allow unlimited searches and interpret deadlines identically across server timezones."""
        self.assertIsNone(Budget().deadline_time())
        self.assertIsNone(deadline_time(None))
        local = Budget("2026-10-24T10:30:00+09:00", 0, 1)
        self.assertEqual(local.deadline_time(), deadline_time("2026-10-24T01:30:00+00:00"))
        self.assertEqual(local.to_dict()["reserve_minutes"], 0)
        for budget in (
            Budget("2026-10-24T10:30:00"),
            Budget("invalid"),
            Budget(finish_by=123),
            Budget(reserve_minutes=-1),
            Budget(reserve_minutes=True),
            Budget(time_safety_factor=0.9),
            Budget(time_safety_factor=math.inf),
        ):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                budget.validate()

    def test_selection_metric_and_ties_use_validation_only(self):
        """Choose the requested primary metric without reading contradictory test values."""
        rows = [
            {"name": "accuracy", "validation": {"accuracy": 0.8, "nll": 1.0}, "test": 0.1},
            {"name": "nll", "validation": {"accuracy": 0.7, "nll": 0.5}, "test": 0.9},
            {"name": "tie", "validation": {"accuracy": 0.8, "nll": 1.0}, "test": 1.0},
        ]
        self.assertIs(select_candidate(rows, "accuracy"), rows[0])
        self.assertIs(select_candidate(rows, "nll"), rows[1])
        for invalid in ([], [{"validation": {"accuracy": math.nan, "nll": 1}}], [{"test": 1}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_candidate(invalid)
        with self.assertRaises(ValueError):
            select_candidate(rows, "test_accuracy")

    def test_stage_order_is_editable_but_test_cannot_precede_training(self):
        """Accept both meaningful training orders and reject missing, duplicate, or early-test stages."""
        for middle in (("tune", "baseline"), ("baseline", "tune")):
            stages = ["preflight", *middle, "evaluate"]
            self.assertEqual(validate_stages(stages), tuple(stages))
        for stages in (
            [],
            ["preflight", "tune", "evaluate", "baseline"],
            ["preflight", "tune", "tune", "evaluate"],
            ["tune", "baseline", "evaluate"],
        ):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                validate_stages(stages)


if __name__ == "__main__":
    unittest.main()
