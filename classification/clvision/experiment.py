"""Composable ERM/CUE experiment stages driven entirely by caller-provided recipes."""

import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filelock import FileLock

from cue_routing.io import atomic_json, canonical_hash, read_json, run_lock, seed_everything

from .cue_workflow import verified_metrics
from .data import load_manifest
from .display import write
from .engine import evaluate_run, train, verify_run
from .experiment_artifacts import collect_candidates, promote_winner, split_audit
from .experiment_runtime import (
    DeadlineProgress,
    admit_candidate,
    estimated_run_seconds,
    preflight_search,
)
from .experiment_settings import CUE_FIELDS, Budget, Search, select_candidate, validate_stages
from .identity import source_hash
from .progress import duration
from .workflow import seal_study


class Comparison:
    """Own one dataset/seed comparison; callers compose stages and supply every training setting."""

    def __init__(
        self,
        *,
        root,
        data_root,
        directory,
        config,
        search: Search,
        budget: Budget = Budget(),
        resume=False,
        preflight_steps=10,
    ):
        self.root, self.data_root, self.directory = map(
            lambda p: Path(p).resolve(), (root, data_root, directory)
        )
        self.config = config.validate()
        self.candidates = search.resolve(config)
        self.search = Search(tuple(dict(row) for row in search.candidates), metric=search.metric)
        self.budget = budget.validate()
        self.deadline = self.budget.deadline_time()
        if type(preflight_steps) is not int or preflight_steps < 10:
            raise ValueError("preflight_steps must be an integer of at least 10")
        if type(resume) is not bool:
            raise ValueError("resume must be boolean")
        self.resume, self.preflight_steps = resume, preflight_steps
        self.ready = False
        self.stack = None
        self.baseline_dir = self.directory / "main" / f"seed{config.seed}_erm"
        self.selected_dir = self.directory / "main" / f"seed{config.seed}_cue"

    def describe(self):
        """Resolve a plain-data experiment preview without reading data or initializing CUDA."""
        return {
            "base_config": self.config.to_dict(),
            "search": self.search.to_dict(),
            "resolved_candidates": [config.to_dict() for config in self.candidates],
            "budget": self.budget.to_dict(),
            "preflight_steps": self.preflight_steps,
            "maximum_training_runs": 1 + len(self.candidates),
            "selection_checkpoint": "fixed_final_epoch",
            "test_policy": "only after search closure and both final models complete",
        }

    def __enter__(self):
        """Freeze resolved inputs and acquire process locks before constructing any training stage."""
        if self.stack is not None:
            raise RuntimeError("Comparison is already open")
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
            self.audit = split_audit(self.manifest)
            self.plan = {
                "schema": 2,
                "protocol": self.describe(),
                "source_sha256": source_hash(self.root),
                "dataset_id": self.manifest["dataset_id"],
                "split_audit": self.audit,
                "benchmark_evidence": not self.config.debug,
                "main_seeds": [self.config.seed],
                "main_methods": ["erm", "cue"],
                "natural_datasets": [self.config.dataset],
                "mechanism_seeds": [],
                "mechanism_methods": [],
            }
            if (self.directory / "plan.json").exists() and not self.resume:
                raise FileExistsError("Study exists; use a new directory or explicit resume")
            plan_path = self.directory / "plan.json"
            if plan_path.exists() and canonical_hash(read_json(plan_path)) != canonical_hash(
                self.plan
            ):
                raise ValueError("Study scope or source changed; use a new experiment directory")
            atomic_json(self.directory / "plan.json", self.plan)
            self.plan_id = canonical_hash(self.plan)
            self.journal_path = self.directory / "search.json"
            self.selection_path = self.directory / "selection.json"
            self.journal = (
                read_json(self.journal_path)
                if self.journal_path.exists()
                else {"plan_sha256": self.plan_id, "decisions": []}
            )
            if self.journal["plan_sha256"] != self.plan_id or len(self.journal["decisions"]) > len(
                self.candidates
            ):
                raise ValueError("Search journal differs from registered plan")
            self.stack = stack
            write(
                f"[split] {self.config.dataset} counts={self.audit['counts']} disjoint=True test=sealed seed={self.config.seed}"
            )
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, *args):
        """Release device and run locks even when a stage fails."""
        self.ready = False
        stack, self.stack = self.stack, None
        return stack.__exit__(*args)

    def _require_open(self, preflight=False):
        """Reject unscoped calls and long training before successful preflight."""
        if self.stack is None:
            raise RuntimeError("Use Comparison inside a with statement")
        if preflight and not self.ready:
            raise RuntimeError("Call preflight() before training")

    def _record(self, stage):
        """Record successful stages without changing training or selection identities."""
        path = self.directory / "stages.json"
        events = read_json(path) if path.exists() else []
        atomic_json(path, events + [{"stage": stage, "time": time.time()}])

    def _baseline_complete(self):
        """Verify a completed baseline before treating its remaining time as zero."""
        if not (self.baseline_dir / "completion.json").exists():
            return False
        verify_run(
            self.baseline_dir, self.root, self.config, self.manifest["dataset_id"], self.plan_id
        )
        return True

    def preflight(self):
        """Check every candidate and the baseline before either full training stage can run."""
        self._require_open()
        seed_everything(self.config.seed, self.config.device)
        erm, cues = preflight_search(
            self.root,
            self.directory,
            self.manifest,
            self.config,
            self.candidates,
            steps=self.preflight_steps,
        )
        self.erm_estimate = estimated_run_seconds(self.config, self.manifest, erm, "erm")
        self.cue_estimates = [
            estimated_run_seconds(config, self.manifest, profile, "cue")
            for config, profile in zip(self.candidates, cues)
        ]
        if not self.journal["decisions"] and not self._baseline_complete():
            if not admit_candidate(
                time.time(),
                self.deadline,
                self.erm_estimate + self.cue_estimates[0],
                self.budget.reserve_minutes,
                self.budget.time_safety_factor,
            ):
                raise RuntimeError(
                    "Budget cannot fit one full CUE trial plus ERM; no long training or test started"
                )
        self.ready = True
        self._record("preflight")

    def baseline(self):
        """Train ERM at the caller's exact recipe, before or after CUE tuning."""
        self._require_open(preflight=True)
        with DeadlineProgress(self.directory, self.deadline, self.config.epochs):
            result = train(
                self.config,
                self.manifest,
                self.baseline_dir,
                self.root,
                selection_id=self.plan_id,
                resume=self.baseline_dir.exists(),
            )
        self._record("baseline")
        return result

    def tune(self):
        """Train declared CUE trials within budget, then freeze the validation-only winner."""
        self._require_open(preflight=True)
        if not self.selection_path.exists():
            baseline_complete = self._baseline_complete()
            slowdown = 1.0
            if baseline_complete:
                baseline_seconds = read_json(self.baseline_dir / "summary.json")["elapsed_seconds"]
                slowdown = max(slowdown, baseline_seconds / max(self.erm_estimate, 1e-9))
            with DeadlineProgress(self.directory, self.deadline, self.config.epochs):
                for index, candidate in enumerate(self.candidates):
                    estimate = self.cue_estimates[index] * slowdown
                    reserved = 0 if baseline_complete else self.erm_estimate * slowdown
                    if index >= len(self.journal["decisions"]):
                        now = time.time()
                        admitted = admit_candidate(
                            now,
                            self.deadline,
                            estimate + reserved,
                            self.budget.reserve_minutes,
                            self.budget.time_safety_factor,
                        )
                        self.journal["decisions"].append(
                            {
                                "index": index,
                                "action": "run" if admitted else "skip_budget",
                                "decided_at": now,
                                "estimated_seconds": estimate,
                                "reserved_erm_seconds": reserved,
                            }
                        )
                        atomic_json(self.journal_path, self.journal)
                    decision = self.journal["decisions"][index]
                    if decision["index"] != index or decision["action"] not in (
                        "run",
                        "skip_budget",
                    ):
                        raise ValueError("Invalid candidate admission journal")
                    write(
                        f"[search] candidate={index + 1}/{len(self.candidates)} action={decision['action']} config={self.search.candidates[index]} estimated={duration(estimate)} erm_reserved={duration(reserved)}"
                    )
                    if decision["action"] == "skip_budget":
                        continue
                    folder = self.directory / "development" / f"cue{index}"
                    result = train(
                        candidate,
                        self.manifest,
                        folder,
                        self.root,
                        phase="development",
                        selection_id=self.plan_id,
                        resume=folder.exists(),
                    )
                    slowdown = max(
                        slowdown, result["elapsed_seconds"] / max(self.cue_estimates[index], 1e-9)
                    )
        selection = self._selection()
        self._record("tune")
        return selection

    def _selection(self):
        """Recompute the frozen choice from verified final validation artifacts, never test scores."""
        rows = collect_candidates(
            self.root,
            self.directory,
            self.manifest,
            self.candidates,
            self.journal["decisions"],
            self.plan_id,
        )
        if len(self.journal["decisions"]) != len(self.candidates) or not rows:
            raise RuntimeError(
                "Search is not closed with a completed candidate; test remains sealed"
            )
        chosen = select_candidate(rows, metric=self.search.metric)
        selection = {
            "plan_sha256": self.plan_id,
            "search": self.journal,
            "candidates": rows,
            "selected": chosen,
            "metric": self.search.metric,
            "selection_data": "validation_only_fixed_final_epoch",
        }
        if self.selection_path.exists() and read_json(self.selection_path) != selection:
            raise ValueError("Frozen validation selection or search journal changed")
        atomic_json(self.selection_path, selection)
        promote_winner(
            self.root,
            self.directory / "development" / chosen["name"],
            self.selected_dir,
            self.candidates[chosen["index"]],
            self.manifest,
            canonical_hash(selection),
        )
        return selection

    def evaluate(self):
        """Open held-out test only after the search and both matched final models are complete."""
        self._require_open()
        if not self.selection_path.exists() or not self._baseline_complete():
            raise RuntimeError(
                "Complete tune() and baseline() before evaluate(); test remains sealed"
            )
        selection = self._selection()
        selected = self.candidates[selection["selected"]["index"]]
        specs = [
            (self.baseline_dir, self.config, self.plan_id, "main"),
            (self.selected_dir, selected, canonical_hash(selection), "main"),
        ]
        contracts = [
            verify_run(folder, self.root, config, self.manifest["dataset_id"], sid)[0]
            for folder, config, sid, _ in specs
        ]
        if len({row["initial_shared_model_sha256"] for row in contracts}) != 1:
            raise ValueError("ERM and CUE initial weights differ")
        common = [
            {k: v for k, v in cfg.to_dict().items() if k not in CUE_FIELDS | {"method"}}
            for _, cfg, _, _ in specs
        ]
        if common[0] != common[1]:
            raise ValueError("ERM and CUE common settings differ")
        seal = seal_study(
            self.directory,
            self.root,
            {**self.plan, "selection_sha256": canonical_hash(selection)},
            specs,
            {self.config.dataset: self.manifest},
        )
        manifest = load_manifest(self.manifest_path, root_override=self.data_root)
        if manifest["dataset_id"] != self.plan["dataset_id"]:
            raise ValueError("Dataset changed before final test")
        metrics = {}
        for spec in specs:
            evaluate_run(manifest, spec[0], self.root, seal)
            metrics[spec[1].method] = verified_metrics(self.root, manifest, spec, seal)
        output = {
            "status": "complete",
            "protocol": self.describe(),
            "split_audit": self.audit,
            "selection": selection,
            "test_metrics": metrics,
            "cue_minus_erm_pp": 100 * (metrics["cue"]["accuracy"] - metrics["erm"]["accuracy"]),
            "study_seal_sha256": canonical_hash(seal),
            "benchmark_evidence": not self.config.debug,
            "finished_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "deadline_met": None if self.deadline is None else time.time() <= self.deadline,
        }
        atomic_json(self.directory / "comparison.json", output)
        lines = [
            "# ERM / CUE comparison",
            "",
            f"Dataset: {self.config.dataset}; seed: {self.config.seed}.",
            f"Split counts: {self.audit['counts']}.",
            f"Source partition policy: {self.audit['source_split_policy']}.",
            "Synthetic debug results; not benchmark evidence."
            if self.config.debug
            else "Selection uses validation only; final test follows training and frozen selection.",
            "",
            "| Method | Final test accuracy |",
            "|---|---:|",
        ]
        lines += [f"| {method} | {100 * row['accuracy']:.2f}% |" for method, row in metrics.items()]
        lines += [
            "",
            f"CUE - ERM: {output['cue_minus_erm_pp']:+.2f} pp.",
            "One training seed per comparison; no statistical significance claim.",
        ]
        (self.directory / "comparison.md").write_text("\n".join(lines) + "\n")
        atomic_json(
            self.directory / "progress.json",
            {
                "stage": "complete",
                "final_test_accessed": True,
                "deadline_met": output["deadline_met"],
            },
        )
        self._record("evaluate")
        write(
            f"[complete] {self.config.dataset} ERM test={100 * metrics['erm']['accuracy']:.2f}% CUE test={100 * metrics['cue']['accuracy']:.2f}% results={self.directory / 'comparison.md'}"
        )
        return output

    def run(self, stages=("preflight", "tune", "baseline", "evaluate")):
        """Execute a caller-ordered pipeline; swapping tune/baseline requires no library edit."""
        self._require_open()
        stages = validate_stages(stages)
        path = self.directory / "execution.json"
        declared = {"stages": list(stages)}
        if path.exists() and read_json(path) != declared:
            raise ValueError("Execution order changed; use a new experiment directory")
        atomic_json(path, declared)
        result = None
        for name in stages:
            result = getattr(self, name)()
        return result
