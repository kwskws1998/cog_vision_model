"""Real-data preflight and measured timing shared by configurable experiment stages."""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from cue_routing.io import atomic_json, read_json

from .display import write
from .profile import profile_dataset, verify_profile
from .progress import CURRENT_PROGRESS, duration


class DeadlineProgress:
    """Report measured run completion times without interpreting validation as test accuracy."""

    def __init__(self, directory, deadline, epochs):
        self.directory, self.deadline, self.epochs = directory, deadline, epochs
        self.seconds = {}

    def __enter__(self):
        self.token = CURRENT_PROGRESS.set(self)
        return self

    def __exit__(self, *args):
        CURRENT_PROGRESS.reset(self.token)

    def update(self, run_dir, event, **values):
        """Use whole epoch measurements including validation and checkpoint overhead."""
        if event != "train_epoch":
            return
        key = str(run_dir)
        rates = self.seconds.setdefault(key, [])
        rates.append(values["epoch_seconds"])
        remaining = sum(rates[-10:]) / len(rates[-10:]) * (self.epochs - values["epoch"])
        now = time.time()
        finish = datetime.fromtimestamp(now + remaining, ZoneInfo("Asia/Seoul")).isoformat()
        status = {
            "active": key,
            "epoch": values["epoch"],
            "run_remaining_seconds": remaining,
            "run_finish_KST_estimate": finish,
            "deadline_remaining_seconds": None if self.deadline is None else self.deadline - now,
            "final_test_accessed": False,
        }
        atomic_json(self.directory / "progress.json", status)
        write(
            f"[budget] run_finish_KST~={finish} deadline_remaining={duration(self.deadline - now) if self.deadline is not None else 'unlimited'} test=sealed"
        )


def estimated_run_seconds(config, manifest, profile, method):
    """Estimate train plus validation from preflight, never from test throughput or test scores."""
    if not profile:
        return 1.0
    row = profile["methods"][method]
    epoch = len(manifest["splits"]["train"]) / row["training_images_per_second"]
    epoch += len(manifest["splits"]["validation"]) / row["evaluation_images_per_second"]
    epoch += row["training_loader_startup_seconds"] + row["evaluation_loader_startup_seconds"]
    return config.epochs * epoch


def admit_candidate(now, deadline, estimate, reserve_minutes, factor):
    """Make a timing-only admission decision; never shorten a candidate's training schedule."""
    return deadline is None or now + estimate * factor + reserve_minutes * 60 <= deadline


def preflight_search(root, directory, manifest, config, candidates, steps=10):
    """Exercise every exact CUE candidate before any long run, then measure the ERM reserve."""
    if config.debug:
        return None, [None] * len(candidates)
    profiles = []
    jobs = [(f"cue{index}", candidate) for index, candidate in enumerate(candidates)]
    jobs.append(("erm", config))
    for name, recipe in jobs:
        path = directory / "preflight" / f"{name}.json"
        write(
            f"[preflight-check] {name} method={recipe.method} tau={recipe.gate_tau} temperature={recipe.gate_temperature} floor={recipe.joint_floor} train/validation_only=True"
        )
        if path.exists():
            profile = read_json(path)
        else:
            profile = profile_dataset(
                root, manifest, recipe, path, steps=steps, methods=(recipe.method,)
            )
        verify_profile(profile, root, manifest, recipe, methods=(recipe.method,))
        if profile.get("active", {}).get("config") != recipe.to_dict():
            raise ValueError(f"Preflight {name} differs from the exact registered candidate config")
        if profile.get("optimizer_steps_per_method") != steps:
            raise ValueError("Preflight step count differs from the registered plan")
        profiles.append(profile)
    measurements = [
        profile["methods"][recipe.method] for profile, (_, recipe) in zip(profiles, jobs)
    ]
    for key in ("initial_shared_model_sha256", "first_augmented_batch_sha256"):
        if len({row[key] for row in measurements}) != 1:
            raise ValueError("CUE candidate and ERM preflights are not matched")
    write(f"[preflight-check] passed cue_candidates={len(candidates)} erm=1")
    return profiles[-1], profiles[:-1]
