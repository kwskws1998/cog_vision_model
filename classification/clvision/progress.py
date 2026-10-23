"""Measured study timing and resumable progress, independent of model selection and metrics."""

import math
import statistics
import time
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cue_routing.io import atomic_json, read_json

CURRENT_PROGRESS = ContextVar("classification_progress", default=None)


def duration(seconds):
    """Format measured or estimated seconds without presenting unknown time as zero."""
    if seconds is None:
        return "estimating"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def progress_event(run_dir, event, **values):
    """Send diagnostics only when a whole-study progress context is active."""
    tracker = CURRENT_PROGRESS.get()
    if tracker is not None:
        tracker.update(run_dir, event, **values)


def planned_jobs(plan):
    """Enumerate the declared development, main, and mechanism jobs without selecting candidates."""
    jobs = {}
    for dataset, trials in plan["development_trials"].items():
        for trial in trials:
            jobs[f"development/{dataset}/{trial['name']}"] = (trial["config"], "development")
    for dataset, base in plan["resolved_recipes"].items():
        mechanism = dataset == "dominoes"
        methods = plan["mechanism_methods"] if mechanism else plan["main_methods"]
        seeds = plan["mechanism_seeds"] if mechanism else plan["main_seeds"]
        for seed in seeds:
            for method in methods:
                config = {**base, "method": method, "seed": seed}
                jobs[f"main/{dataset}/seed{seed}_{method}"] = (
                    config,
                    "mechanism" if mechanism else "main",
                )
    return jobs


class StudyProgress:
    """Estimate the remaining declared work using per-dataset/method measurements on this host."""

    def __init__(self, directory, plan, manifests, profiles=None, started=None):
        """Recover completed work from artifacts and seed provisional rates from CUDA preflight."""
        self.directory = Path(directory).resolve()
        self.started = time.monotonic() if started is None else started
        self.rates, self.jobs = {}, {}
        self.status, self.stage, self.active = "running", "training", None
        self.last_emit = -math.inf
        self.active_batch = None
        for key, (config, phase) in planned_jobs(plan).items():
            dataset, method = config["dataset"], config["method"]
            splits = manifests[dataset]["splits"]
            train_n = len(splits["train"])
            if config["debug_max_batches"]:
                train_n = min(train_n, config["debug_max_batches"] * config["batch_size"])
            test_n = len(splits["test"])
            evaluation_n = (
                0 if phase == "development" else test_n * (6 if dataset == "dominoes" else 1)
            )
            corruption_n = (
                75 if dataset == "cifar100" and phase == "main" and not config["debug"] else 0
            )
            folder = self.directory / key
            history = (
                read_json(folder / "history.json") if (folder / "history.json").exists() else []
            )
            epochs_done = history[-1]["epoch"] if history else 0
            complete = (folder / "completion.json").exists()
            cells = (
                read_json(folder / "corruption_cells.json").get("cells", {})
                if (folder / "corruption_cells.json").exists()
                else {}
            )
            self.jobs[key] = {
                "dataset": dataset,
                "method": method,
                "seed": config["seed"],
                "phase": phase,
                "epochs": config["epochs"],
                "epochs_done": config["epochs"] if complete else epochs_done,
                "complete": complete,
                "train_images": train_n,
                "validation_images": len(splits["validation"]),
                "clean_images": evaluation_n,
                "clean_done": (folder / "test_metrics.json").exists(),
                "corruption_cells": corruption_n,
                "corruption_done": min(corruption_n, len(cells)),
                "cell_images": test_n,
            }
            rate_key = (dataset, method)
            if rate_key not in self.rates:
                profile = ((profiles or {}).get(dataset) or {}).get("methods", {}).get(method, {})
                self.rates[rate_key] = {
                    "train_images_per_second": profile.get("training_images_per_second"),
                    "eval_images_per_second": profile.get("evaluation_images_per_second"),
                    "epochs": [],
                }
            if history and history[-1].get("elapsed_seconds", 0) > 0 and epochs_done > 0:
                self.rates[rate_key]["epochs"].append(history[-1]["elapsed_seconds"] / epochs_done)

    def __enter__(self):
        """Publish the initial estimate before the first full training run."""
        self.token = CURRENT_PROGRESS.set(self)
        self.emit(force=True)
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Persist completion or failure while leaving the original exception intact."""
        CURRENT_PROGRESS.reset(self.token)
        self.status = "failed" if exception_type else "complete"
        self.stage = self.status
        self.emit(force=True, error=str(exception) if exception_type else None)

    def update(self, run_dir, event, **values):
        """Refine timing from accepted epochs and completed evaluations without changing the recipe."""
        key = str(Path(run_dir).resolve().relative_to(self.directory))
        job = self.jobs[key]
        rate = self.rates[(job["dataset"], job["method"])]
        self.active = key
        self.stage = "training" if event.startswith("train") else "final_evaluation"
        if event == "train_start":
            job["epochs_done"] = values["completed_epochs"]
            self.active_batch = None
        elif event == "train_batch":
            self.active_batch = values
        elif event == "train_epoch":
            job["epochs_done"] = values["epoch"]
            self.active_batch = None
            if values["epoch_seconds"] > 0:
                rate["epochs"] = (rate["epochs"] + [values["epoch_seconds"]])[-5:]
            rate["train_epoch_seconds"] = values["training_seconds"]
            if values["validation_seconds"] > 0:
                rate["eval_images_per_second"] = (
                    job["validation_images"] / values["validation_seconds"]
                )
        elif event == "train_complete":
            job.update(epochs_done=job["epochs"], complete=True)
            self.active_batch = None
        elif event == "clean_complete":
            job["clean_done"] = True
            if values.get("seconds", 0) > 0:
                rate["eval_images_per_second"] = job["clean_images"] / values["seconds"]
        elif event == "corruption_cell":
            job["corruption_done"] = values["completed_cells"]
            if values.get("seconds", 0) > 0:
                rate["corruption_images_per_second"] = job["cell_images"] / values["seconds"]
        self.emit(force=event != "train_batch")

    def estimate(self):
        """Sum remaining training, validation, clean diagnostics, and corruption evaluation work."""
        training, evaluation, unknown = 0.0, 0.0, set()
        for key, job in self.jobs.items():
            rate = self.rates[(job["dataset"], job["method"])]
            remaining_epochs = max(0, job["epochs"] - job["epochs_done"])
            eval_rate = rate["eval_images_per_second"]
            if remaining_epochs:
                epoch_seconds = None
                if rate["epochs"]:
                    epoch_seconds = statistics.mean(rate["epochs"][-5:])
                elif rate["train_images_per_second"] and eval_rate:
                    epoch_seconds = (
                        job["train_images"] / rate["train_images_per_second"]
                        + job["validation_images"] / eval_rate
                    )
                else:
                    unknown.add(f"{job['dataset']}/{job['method']}:training")
                if epoch_seconds is not None:
                    pending = remaining_epochs * epoch_seconds
                    if key == self.active and self.active_batch:
                        train_seconds = rate.get("train_epoch_seconds")
                        if train_seconds is None and rate["train_images_per_second"]:
                            train_seconds = job["train_images"] / rate["train_images_per_second"]
                        if train_seconds is not None:
                            fraction = min(
                                1, self.active_batch["batch"] / self.active_batch["batches"]
                            )
                            pending -= min(train_seconds, epoch_seconds) * fraction
                    training += max(0, pending)
            if job["clean_images"] and not job["clean_done"]:
                if eval_rate:
                    evaluation += job["clean_images"] / eval_rate
                else:
                    unknown.add(f"{job['dataset']}/{job['method']}:evaluation")
            cells = job["corruption_cells"] - job["corruption_done"]
            if cells:
                corruption_rate = rate.get("corruption_images_per_second") or eval_rate
                if corruption_rate:
                    evaluation += cells * job["cell_images"] / corruption_rate
                else:
                    unknown.add(f"{job['dataset']}/{job['method']}:corruptions")
        return training, evaluation, sorted(unknown)

    def emit(self, force=False, error=None):
        """Write a current progress snapshot and a human-readable line at least every 30 training seconds."""
        now = time.monotonic()
        if not force and now - self.last_emit < 30:
            return
        self.last_emit = now
        training, evaluation, unknown = self.estimate()
        remaining = None if unknown else training + evaluation
        if self.status == "complete":
            remaining = 0.0
        timestamp = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
        finish = (
            (timestamp + timedelta(seconds=remaining)).isoformat(timespec="seconds")
            if remaining is not None and self.status != "failed"
            else None
        )
        done = sum(job["complete"] for job in self.jobs.values())
        clean_jobs = [job for job in self.jobs.values() if job["clean_images"]]
        clean_done = sum(job["clean_done"] for job in clean_jobs)
        cells_done = sum(job["corruption_done"] for job in self.jobs.values())
        cells_total = sum(job["corruption_cells"] for job in self.jobs.values())
        payload = {
            "schema": 1,
            "status": self.status,
            "stage": self.stage,
            "updated_at_kst": timestamp.isoformat(),
            "session_elapsed_seconds": now - self.started,
            "completed_training_runs": done,
            "total_training_runs": len(self.jobs),
            "completed_clean_evaluations": clean_done,
            "total_clean_evaluations": len(clean_jobs),
            "completed_corruption_cells": cells_done,
            "total_corruption_cells": cells_total,
            "active_run": self.active,
            "active_batch": self.active_batch,
            "estimated_training_remaining_seconds": training
            if not any(":training" in key for key in unknown)
            else None,
            "estimated_evaluation_remaining_seconds": evaluation
            if not any(":training" not in key for key in unknown)
            else None,
            "estimated_remaining_seconds": remaining,
            "estimated_finish_kst": finish,
            "unmeasured_rates": unknown,
            "timing_basis": "Per-dataset/method preflight throughput, refined by completed epochs and evaluations; provisional until measured. Epoch observations include validation/checkpoint writes. Startup, final sealing, reporting, and downtime are not predicted.",
            "jobs": self.jobs,
            "error": error,
        }
        atomic_json(self.directory / "progress.json", payload)
        print(
            f"[study] stage={self.stage} runs={done}/{len(self.jobs)} "
            f"clean={clean_done}/{len(clean_jobs)} corruptions={cells_done}/{cells_total} "
            f"elapsed={duration(now - self.started)} remaining~={duration(remaining)} "
            f"finish_KST~={finish or 'estimating'} active={self.active or 'none'}",
            flush=True,
        )
