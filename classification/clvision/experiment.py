"""Composable ERM/CUE experiment stages driven entirely by caller-provided recipes."""

import copy
import time
from contextlib import ExitStack
from dataclasses import replace
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

from .config import Config
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
        include_baseline=True,
        reference=None,
        experiment_source=None,
    ):
        self.root, self.data_root, self.directory = map(
            lambda p: Path(p).resolve(), (root, data_root, directory)
        )
        self.config = config.validate()
        if type(include_baseline) is not bool:
            raise ValueError("include_baseline must be boolean")
        if config.method != ("erm" if include_baseline else "cue"):
            raise ValueError(
                "Use method=erm for paired comparison or method=cue for CUE-only search"
            )
        self.include_baseline = include_baseline
        self.reference = copy.deepcopy(reference)
        self.experiment_source = None if experiment_source is None else Path(experiment_source)
        if reference is not None:
            if include_baseline:
                raise ValueError("Historical validation reference is for CUE-only search")
            required = {
                "config",
                "validation",
                "dataset_id",
                "initial_shared_model_sha256",
                "provenance",
            }
            if not isinstance(reference, dict) or set(reference) != required:
                raise ValueError(
                    "Reference requires config, validation, dataset/initial hashes and provenance"
                )
            previous = Config.from_dict(reference["config"])
            common = [
                {k: v for k, v in cfg.to_dict().items() if k not in CUE_FIELDS}
                for cfg in (previous, config)
            ]
            if previous.method != "cue" or canonical_hash(common[0]) != canonical_hash(common[1]):
                raise ValueError("Reference and new candidates must share common settings and seed")
            select_candidate([reference], metric=search.metric)
            for key in ("dataset_id", "initial_shared_model_sha256"):
                value = reference[key]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise ValueError(f"Invalid reference {key}")
            if (
                type(reference["validation"].get("n")) is not int
                or reference["validation"]["n"] <= 0
            ):
                raise ValueError("Reference must record its validation sample count")
            if not 0 <= reference["validation"]["accuracy"] <= 1:
                raise ValueError("Reference accuracy must be a fraction in [0, 1]")
            if not isinstance(reference["provenance"], dict) or not reference["provenance"]:
                raise ValueError("Reference must record its provenance")
        self.candidates = search.resolve(replace(config, method="erm"))
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
            "maximum_training_runs": int(self.include_baseline) + len(self.candidates),
            "include_baseline": self.include_baseline,
            "reference": copy.deepcopy(self.reference),
            "reference_tie_policy": "keep reference unless primary validation metric strictly improves",
            "experiment_source_sha256": None
            if self.experiment_source is None
            else file_hash(self.experiment_source),
            "selection_checkpoint": "fixed_final_epoch",
            "test_policy": "after search closure; only the selected new CUE and optional ERM; no new test if reference retained",
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
            if self.reference is not None and (
                self.reference["dataset_id"] != self.manifest["dataset_id"]
                or self.reference["validation"]["n"] != self.audit["counts"]["validation"]
            ):
                raise ValueError("Reference dataset or validation count differs")
            self.plan = {
                "schema": 2,
                "protocol": self.describe(),
                "source_sha256": source_hash(self.root),
                "dataset_id": self.manifest["dataset_id"],
                "split_audit": self.audit,
                "benchmark_evidence": not self.config.debug,
                "main_seeds": [self.config.seed],
                "main_methods": ["erm", "cue"] if self.include_baseline else ["cue"],
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
            if self.experiment_source is not None:
                snapshot = self.directory / "experiment.py"
                if snapshot.exists() and file_hash(snapshot) != file_hash(self.experiment_source):
                    raise ValueError("Experiment snapshot differs from registered source")
                if not snapshot.exists():
                    snapshot.write_bytes(self.experiment_source.read_bytes())
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
        if (
            canonical_hash(self.describe()) != canonical_hash(self.plan["protocol"])
            or source_hash(self.root) != self.plan["source_sha256"]
        ):
            raise ValueError("Study inputs or source changed after registration")
        if preflight and not self.ready:
            raise RuntimeError("Call preflight() before training")

    def _record(self, stage):
        """Record successful stages without changing training or selection identities."""
        path = self.directory / "stages.json"
        events = read_json(path) if path.exists() else []
        atomic_json(path, events + [{"stage": stage, "time": time.time()}])

    def _baseline_complete(self):
        """Verify a completed baseline before treating its remaining time as zero."""
        if not self.include_baseline:
            return True
        if not (self.baseline_dir / "completion.json").exists():
            return False
        verify_run(
            self.baseline_dir, self.root, self.config, self.manifest["dataset_id"], self.plan_id
        )
        return True

    def preflight(self):
        """Check every candidate and the optional baseline before full training can run."""
        self._require_open()
        self.ready = False
        seed_everything(self.config.seed, self.config.device)
        options = {} if self.include_baseline else {"include_baseline": False}
        erm, cues = preflight_search(
            self.root,
            self.directory,
            self.manifest,
            self.config,
            self.candidates,
            steps=self.preflight_steps,
            **options,
        )
        if self.reference is not None and not self.config.debug:
            if any(
                row["methods"]["cue"]["initial_shared_model_sha256"]
                != self.reference["initial_shared_model_sha256"]
                for row in cues
            ):
                raise ValueError("Candidate initial weights differ from historical reference")
        self.erm_estimate = (
            estimated_run_seconds(self.config, self.manifest, erm, "erm")
            if self.include_baseline
            else 0.0
        )
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
        write(
            f"[estimate] planned_training_runs={int(self.include_baseline) + len(self.candidates)} planned_train+validation~={duration(self.erm_estimate + sum(self.cue_estimates))}"
        )
        self._record("preflight")

    def baseline(self):
        """Train ERM at the caller's exact recipe, before or after CUE tuning."""
        self._require_open(preflight=True)
        if not self.include_baseline:
            raise RuntimeError("Baseline training is disabled for this search")
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
            if self.include_baseline and baseline_complete:
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
        reference_retained = False
        if self.reference is not None:
            for row in rows:
                contract = read_json(self.directory / "development" / row["name"] / "contract.json")
                if (
                    contract["initial_shared_model_sha256"]
                    != self.reference["initial_shared_model_sha256"]
                ):
                    raise ValueError("Candidate initial weights differ from historical reference")
            score = chosen["validation"][self.search.metric]
            previous_score = self.reference["validation"][self.search.metric]
            reference_retained = (
                score <= previous_score
                if self.search.metric == "accuracy"
                else score >= previous_score
            )
        selection = {
            "plan_sha256": self.plan_id,
            "search": self.journal,
            "candidates": rows,
            "selected": None if reference_retained else chosen,
            "best_new_candidate": chosen,
            "reference": copy.deepcopy(self.reference),
            "reference_retained": reference_retained,
            "metric": self.search.metric,
            "selection_data": "validation_only_fixed_final_epoch",
        }
        if self.selection_path.exists() and read_json(self.selection_path) != selection:
            raise ValueError("Frozen validation selection or search journal changed")
        atomic_json(self.selection_path, selection)
        if not reference_retained:
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
        """Test a new validation winner only after search closure and optional baseline completion."""
        self._require_open()
        if not self.selection_path.exists() or not self._baseline_complete():
            stages = "tune() and baseline()" if self.include_baseline else "tune()"
            raise RuntimeError(f"Complete {stages} before evaluate(); test remains sealed")
        selection = self._selection()
        if selection["reference_retained"]:
            if self.selected_dir.exists():
                raise ValueError("Retained reference unexpectedly has a newly promoted checkpoint")
            return self._report(selection, {}, None)
        selected = self.candidates[selection["selected"]["index"]]
        specs = [(self.selected_dir, selected, canonical_hash(selection), "main")]
        if self.include_baseline:
            specs.insert(0, (self.baseline_dir, self.config, self.plan_id, "main"))
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
        if any(row != common[0] for row in common):
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
        return self._report(selection, metrics, seal)

    def _report(self, selection, metrics, seal):
        """Report validation decisions separately from optional newly opened final-test results."""
        output = {
            "status": "complete",
            "protocol": self.describe(),
            "split_audit": self.audit,
            "selection": selection,
            "test_metrics": metrics,
            "cue_minus_erm_pp": 100 * (metrics["cue"]["accuracy"] - metrics["erm"]["accuracy"])
            if "erm" in metrics
            else None,
            "study_seal_sha256": None if seal is None else canonical_hash(seal),
            "final_test_accessed": bool(metrics),
            "benchmark_evidence": not self.config.debug,
            "finished_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "deadline_met": None if self.deadline is None else time.time() <= self.deadline,
        }
        atomic_json(self.directory / "comparison.json", output)
        lines = [
            "# ERM / CUE comparison" if self.include_baseline else "# CUE validation search",
            "",
            f"Dataset: {self.config.dataset}; seed: {self.config.seed}.",
            f"Split counts: {self.audit['counts']}.",
            f"Source partition policy: {self.audit['source_split_policy']}.",
            "Synthetic debug results; not benchmark evidence."
            if self.config.debug
            else "Selection uses validation only; final test follows training and frozen selection.",
            "",
            "| Candidate | Tau | Temperature | Floor | Final validation accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        rows = list(selection["candidates"])
        if self.reference is not None:
            rows.insert(0, {"name": "historical reference", **self.reference})
        for row in rows:
            cfg = row["config"]
            lines.append(
                f"| {row['name']} | {cfg['gate_tau']} | {cfg['gate_temperature']} | {cfg['joint_floor']} | {100 * row['validation']['accuracy']:.2f}% |"
            )
        if selection["reference_retained"]:
            lines += ["", "Historical reference retained; no new test evaluation or retraining."]
        else:
            lines += ["", f"Validation-selected new candidate: {selection['selected']['name']}."]
        lines += ["", "| Method | Final test accuracy |", "|---|---:|"]
        lines += [f"| {method} | {100 * row['accuracy']:.2f}% |" for method, row in metrics.items()]
        if not metrics:
            lines.append("| Not evaluated | — |")
        if "erm" in metrics:
            lines += ["", f"CUE - ERM: {output['cue_minus_erm_pp']:+.2f} pp."]
        lines += [
            "",
            "One training seed per comparison; no statistical significance claim.",
        ]
        (self.directory / "comparison.md").write_text("\n".join(lines) + "\n")
        atomic_json(
            self.directory / "progress.json",
            {
                "stage": "complete",
                "final_test_accessed": bool(metrics),
                "deadline_met": output["deadline_met"],
            },
        )
        self._record("evaluate")
        write(
            f"[complete] {self.config.dataset} reference_retained={selection['reference_retained']} final_test_accessed={bool(metrics)} results={self.directory / 'comparison.md'}"
        )
        return output

    def run(self, stages=None):
        """Execute a caller-ordered pipeline; swapping tune/baseline requires no library edit."""
        self._require_open()
        if stages is None:
            stages = (
                ("preflight", "tune", "baseline", "evaluate")
                if self.include_baseline
                else ("preflight", "tune", "evaluate")
            )
        stages = validate_stages(stages, include_baseline=self.include_baseline)
        path = self.directory / "execution.json"
        declared = {"stages": list(stages)}
        if path.exists() and read_json(path) != declared:
            raise ValueError("Execution order changed; use a new experiment directory")
        atomic_json(path, declared)
        result = None
        for name in stages:
            result = getattr(self, name)()
        return result
