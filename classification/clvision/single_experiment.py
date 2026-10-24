"""Run one caller-configured control with measured timing and sealed final-test evaluation."""

import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filelock import FileLock

from cue_routing.io import (
    atomic_json,
    canonical_hash,
    file_hash,
    read_json,
    run_lock,
    seed_everything,
)

from .cue_workflow import verified_metrics
from .data import load_manifest
from .display import write
from .engine import evaluate_run, make_model, train, verify_run
from .experiment_artifacts import split_audit
from .experiment_runtime import DeadlineProgress, admit_candidate, estimated_run_seconds
from .experiment_settings import Budget
from .identity import source_hash
from .profile import profile_dataset, verify_profile
from .progress import duration
from .workflow import seal_study


class SingleRun:
    """Compose preflight, training and final evaluation for one explicitly supplied Config."""

    def __init__(
        self,
        *,
        root,
        data_root,
        directory,
        config,
        budget=Budget(),
        resume=False,
        preflight_steps=10,
        experiment_source=None,
        expected_dataset_id=None,
        expected_initial_weights=None,
    ):
        self.root, self.data_root, self.directory = map(
            lambda value: Path(value).resolve(), (root, data_root, directory)
        )
        self.config = config.validate()
        if config.dataset not in ("cifar100", "tiny_imagenet"):
            raise ValueError("SingleRun supports CIFAR-100 and TinyImageNet")
        if type(resume) is not bool or type(preflight_steps) is not int or preflight_steps < 10:
            raise ValueError("resume must be boolean and preflight_steps an integer >= 10")
        self.budget = budget.validate()
        self.deadline = budget.deadline_time()
        self.resume, self.preflight_steps = resume, preflight_steps
        self.experiment_source = None if experiment_source is None else Path(experiment_source)
        self.expected_dataset_id = expected_dataset_id
        self.expected_initial_weights = expected_initial_weights
        self.run_dir = self.directory / "main" / f"seed{config.seed}_{config.method}"
        self.stack, self.ready = None, False

    def describe(self):
        """Preview the exact recipe without initializing CUDA or reading dataset arrays."""
        return {
            "config": self.config.to_dict(),
            "budget": self.budget.to_dict(),
            "preflight_steps": self.preflight_steps,
            "training_runs": 1,
            "tuning": False,
            "checkpoint": "fixed_final_epoch",
            "test_policy": "validation during training; test only after completed training",
            "expected_dataset_id": self.expected_dataset_id,
            "expected_initial_weights": self.expected_initial_weights,
            "experiment_source_sha256": (
                None if self.experiment_source is None else file_hash(self.experiment_source)
            ),
        }

    def __enter__(self):
        """Lock the device and output, audit split isolation, then freeze the complete recipe."""
        if self.stack is not None:
            raise RuntimeError("SingleRun is already open")
        stack = ExitStack()
        try:
            lock = self.root / ".cache/classification_device.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(FileLock(str(lock), timeout=0))
            stack.enter_context(run_lock(self.directory))
            self.manifest_path = self.data_root / "manifests" / f"{self.config.dataset}.json"
            self.manifest = load_manifest(
                self.manifest_path, root_override=self.data_root, defer_test=True
            )
            if self.manifest["dataset"] != self.config.dataset:
                raise ValueError("Dataset manifest differs from requested dataset")
            if self.expected_dataset_id is not None and (
                self.manifest["dataset_id"] != self.expected_dataset_id
            ):
                raise ValueError("Dataset identity differs from the requested reference")
            self.audit = split_audit(self.manifest)
            self.plan = {
                "schema": 1,
                "protocol": self.describe(),
                "source_sha256": source_hash(self.root),
                "dataset_id": self.manifest["dataset_id"],
                "split_audit": self.audit,
                "benchmark_evidence": not self.config.debug,
                "main_seeds": [self.config.seed],
                "main_methods": [self.config.method],
                "natural_datasets": [self.config.dataset],
                "mechanism_seeds": [],
                "mechanism_methods": [],
            }
            path = self.directory / "plan.json"
            if path.exists() and not self.resume:
                raise FileExistsError("Study exists; use a new directory or explicit resume")
            if path.exists() and canonical_hash(read_json(path)) != canonical_hash(self.plan):
                raise ValueError("Study recipe or source changed; use a new experiment directory")
            atomic_json(path, self.plan)
            self.plan_id = canonical_hash(self.plan)
            if self.experiment_source is not None:
                snapshot = self.directory / "experiment.py"
                if snapshot.exists() and file_hash(snapshot) != file_hash(self.experiment_source):
                    raise ValueError("Experiment snapshot differs from registered source")
                if not snapshot.exists():
                    snapshot.write_bytes(self.experiment_source.read_bytes())
            self.stack = stack
            write(
                f"[split] {self.config.dataset} counts={self.audit['counts']} "
                f"disjoint=True test=sealed method={self.config.method} seed={self.config.seed}"
            )
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, *args):
        """Release process locks even when preflight, training or final evaluation fails."""
        self.ready = False
        stack, self.stack = self.stack, None
        return stack.__exit__(*args)

    def _require_open(self):
        """Reject unscoped stages and changes to inputs or executable source after registration."""
        if self.stack is None:
            raise RuntimeError("Use SingleRun inside a with statement")
        if (
            canonical_hash(self.describe()) != canonical_hash(self.plan["protocol"])
            or source_hash(self.root) != self.plan["source_sha256"]
        ):
            raise ValueError("Study recipe or source changed after registration")

    def _check_initial_weights(self, actual):
        """Require the prior model's initial tensors when the caller supplied a reference hash."""
        if self.expected_initial_weights is not None and actual != self.expected_initial_weights:
            raise ValueError("Initial model weights differ from the requested reference")

    def preflight(self):
        """Measure only this exact method on train/validation; failed checks prohibit full training."""
        self._require_open()
        self.ready = False
        seed_everything(self.config.seed, self.config.device)
        profile = None
        if self.config.debug:
            model = make_model(self.config)
            initial = model.shared_state_hash()
            del model
        else:
            path = self.directory / "preflight.json"
            profile = (
                read_json(path)
                if path.exists()
                else profile_dataset(
                    self.root,
                    self.manifest,
                    self.config,
                    path,
                    steps=self.preflight_steps,
                    methods=(self.config.method,),
                )
            )
            verify_profile(
                profile, self.root, self.manifest, self.config, methods=(self.config.method,)
            )
            if canonical_hash(profile.get("active", {}).get("config")) != canonical_hash(
                self.config.to_dict()
            ):
                raise ValueError("Preflight differs from the exact registered config")
            if profile.get("optimizer_steps_per_method") != self.preflight_steps:
                raise ValueError("Preflight step count differs from registered plan")
            initial = profile["methods"][self.config.method]["initial_shared_model_sha256"]
        self._check_initial_weights(initial)
        self.estimate = estimated_run_seconds(
            self.config, self.manifest, profile, self.config.method
        )
        self.ready = True
        write(
            f"[estimate] runs=1 method={self.config.method} train+validation~={duration(self.estimate)} "
            f"measured={not self.config.debug} test=sealed"
        )
        return profile

    def train(self):
        """Train the declared full schedule without invoking any baseline or tuning grid."""
        self._require_open()
        if not self.ready:
            raise RuntimeError("Call preflight() before training")
        complete = (self.run_dir / "completion.json").exists()
        if not complete and not admit_candidate(
            time.time(),
            self.deadline,
            self.estimate,
            self.budget.reserve_minutes,
            self.budget.time_safety_factor,
        ):
            raise RuntimeError("Budget cannot fit the declared run; training and test not started")
        with DeadlineProgress(self.directory, self.deadline, self.config.epochs):
            return train(
                self.config,
                self.manifest,
                self.run_dir,
                self.root,
                phase="main",
                selection_id=self.plan_id,
                resume=self.resume,
            )

    def evaluate(self):
        """Verify full completion before loading test data and reuse verified final predictions on resume."""
        self._require_open()
        if not (self.run_dir / "completion.json").exists():
            raise RuntimeError("Complete train() before evaluate(); test remains sealed")
        contract, _, summary = verify_run(
            self.run_dir, self.root, self.config, self.manifest["dataset_id"], self.plan_id
        )
        self._check_initial_weights(contract["initial_shared_model_sha256"])
        spec = (self.run_dir, self.config, self.plan_id, "main")
        seal = seal_study(
            self.directory, self.root, self.plan, [spec], {self.config.dataset: self.manifest}
        )
        manifest = load_manifest(self.manifest_path, root_override=self.data_root)
        if manifest["dataset_id"] != self.plan["dataset_id"]:
            raise ValueError("Dataset changed before final test")
        evaluate_run(manifest, self.run_dir, self.root, seal)
        metrics = verified_metrics(self.root, manifest, spec, seal)
        output = {
            "status": "complete",
            "protocol": self.plan["protocol"],
            "split_audit": self.audit,
            "final_validation": summary["final_validation"],
            "final_test": metrics,
            "training_elapsed_seconds": summary["elapsed_seconds"],
            "initial_shared_model_sha256": contract["initial_shared_model_sha256"],
            "dataset_id": manifest["dataset_id"],
            "study_seal_sha256": canonical_hash(seal),
            "benchmark_evidence": not self.config.debug,
            "finished_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        }
        atomic_json(self.directory / "results.json", output)
        lines = [
            f"# {self.config.dataset} / {self.config.method} / seed {self.config.seed}",
            "",
            f"Split counts: {self.audit['counts']}. Fixed final epoch: {self.config.epochs}.",
            "Synthetic smoke only; not benchmark evidence."
            if self.config.debug
            else "Validation was evaluated during training; held-out test was evaluated after completion.",
            f"Source partition policy: {self.audit['source_split_policy']}.",
            "",
            "| Metric | Accuracy |",
            "|---|---:|",
            f"| Final validation | {100 * summary['final_validation']['accuracy']:.2f}% |",
            f"| Final test | {100 * metrics['accuracy']:.2f}% |",
            "",
            f"Training + validation elapsed: {duration(summary['elapsed_seconds'])}.",
        ]
        (self.directory / "results.md").write_text("\n".join(lines) + "\n")
        atomic_json(
            self.directory / "progress.json", {"stage": "complete", "final_test_accessed": True}
        )
        write(
            f"[complete] final_test_accuracy={metrics['accuracy']:.4f} results={self.directory / 'results.md'}"
        )
        return output
