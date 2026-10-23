"""Deadline-aware validation search, immutable winner promotion, and two final test evaluations."""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from filelock import FileLock

from cue_routing.io import (
    atomic_json,
    canonical_hash,
    file_hash,
    read_json,
    run_lock,
    seed_everything,
    state_hash,
)

from .ablation_workflow import winner
from .cue_workflow import verified_metrics
from .data import load_manifest
from .display import write
from .engine import evaluate_run, seal_run, train, verify_run
from .identity import source_hash
from .overnight_settings import CUE_FIELDS, deadline_time, preview, validate_settings
from .profile import profile_dataset, verify_profile
from .progress import CURRENT_PROGRESS, duration
from .workflow import check_plan, seal_study


class DeadlineProgress:
    """Report measured run completion times without interpreting validation as test accuracy."""

    def __init__(self, directory, deadline, epochs):
        self.directory, self.deadline, self.epochs = Path(directory), deadline, epochs
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
            "deadline_remaining_seconds": self.deadline - now,
            "final_test_accessed": False,
        }
        atomic_json(self.directory / "progress.json", status)
        write(
            f"[budget] run_finish_KST~={finish} deadline_remaining={duration(self.deadline - now)} test=sealed"
        )


def split_audit(manifest):
    """Distinguish source-qualified identities; equal numeric train/test indices are not overlap."""
    splits = manifest["splits"]
    ids = {
        key: {f"{'test' if key == 'test' else 'train'}:{i}" for i in rows}
        for key, rows in splits.items()
    }
    if any(
        ids[a] & ids[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Train, validation and test overlap")
    counts = {name: len(items) for name, items in ids.items()}
    if manifest["kind"] != "fixture" and counts != {
        "train": 45000,
        "validation": 5000,
        "test": 10000,
    }:
        raise ValueError("Unexpected production split sizes")
    return {
        "counts": counts,
        "pairwise_disjoint": True,
        "identity_hashes": {key: canonical_hash(sorted(items)) for key, items in ids.items()},
        "test_policy": "deferred until search closure; no test-driven tuning",
    }


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
    return now + estimate * factor + reserve_minutes * 60 <= deadline


def collect_candidates(root, directory, manifest, configs, decisions, plan_id):
    """Reconstruct selection from complete, untouched validation-only trials in registered order."""
    rows = []
    for index, decision in enumerate(decisions):
        if decision["index"] != index or decision["action"] not in ("run", "skip_budget"):
            raise ValueError("Invalid candidate admission journal")
        folder = directory / "development" / f"cue{index}"
        if decision["action"] == "skip_budget":
            if folder.exists():
                raise ValueError("Skipped candidate unexpectedly has training artifacts")
            continue
        if (folder / "test_access.json").exists():
            raise ValueError("Tuning candidate accessed final test")
        contract, _, summary = verify_run(
            folder, root, configs[index], manifest["dataset_id"], plan_id
        )
        if contract["phase"] != "development":
            raise ValueError("Candidate must be validation-only development")
        rows.append(
            {
                "name": f"cue{index}",
                "index": index,
                "config": configs[index].to_dict(),
                "validation": summary["final_validation"],
                "completion_sha256": file_hash(folder / "completion.json"),
            }
        )
    return rows


def promote_winner(root, source, destination, config, manifest, selection_id):
    """Copy the frozen winner's final weights, preserving an auditable link without retraining."""
    original, completion, summary = verify_run(source, root, config, manifest["dataset_id"])
    final = torch.load(source / "final.pt", map_location="cpu", weights_only=True)
    contract = {
        **original,
        "phase": "main",
        "development_selection_sha256": selection_id,
        "promotion": {
            "source": str(source.resolve()),
            "completion": completion,
            "model_sha256": state_hash(final["model"]),
            "retrained": False,
        },
    }
    with run_lock(destination):
        if (destination / "completion.json").exists():
            current, _, _ = verify_run(
                destination, root, config, manifest["dataset_id"], selection_id
            )
            copied = torch.load(destination / "final.pt", map_location="cpu", weights_only=True)
            if current != contract or state_hash(copied["model"]) != state_hash(final["model"]):
                raise ValueError("Promoted winner differs from frozen source checkpoint")
        else:
            atomic_json(destination / "contract.json", contract)
            seal_run(
                destination,
                contract,
                final["model"],
                read_json(source / "history.json"),
                summary["accepted_optimizer_steps"],
                summary["elapsed_seconds"],
            )


def run_overnight(root, data_root, directory, settings, finish_by, resume=False):
    """Complete ERM and as many full CUE trials as fit, freeze selection, then open test twice."""
    root, data_root, directory = map(lambda p: Path(p).resolve(), (root, data_root, directory))
    config, candidates = validate_settings(settings)
    deadline = deadline_time(finish_by)
    lock = root / ".cache/classification_device.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0), run_lock(directory):
        path = data_root / "manifests/cifar100.json"
        manifest = load_manifest(path, root_override=data_root, defer_test=True)
        audit = split_audit(manifest)
        plan = {
            "schema": 1,
            "settings": settings,
            "protocol": preview(settings, finish_by),
            "source_sha256": source_hash(root),
            "dataset_id": manifest["dataset_id"],
            "split_audit": audit,
            "benchmark_evidence": not config.debug,
            "main_seeds": [config.seed],
            "main_methods": ["erm", "cue"],
            "natural_datasets": ["cifar100"],
            "mechanism_seeds": [],
            "mechanism_methods": [],
        }
        plan_path = directory / "plan.json"
        if plan_path.exists() and not resume:
            raise FileExistsError("Study exists; use --resume with the same settings and deadline")
        check_plan(directory, plan)
        atomic_json(plan_path, plan)
        plan_id = canonical_hash(plan)
        write(
            f"[split] train={audit['counts']['train']} validation={audit['counts']['validation']} test={audit['counts']['test']} disjoint=True test=sealed seed={config.seed}"
        )
        baseline = directory / "main" / f"seed{config.seed}_erm"
        selected = directory / "main" / f"seed{config.seed}_cue"
        journal_path, selection_path = directory / "search.json", directory / "selection.json"
        journal = (
            read_json(journal_path)
            if journal_path.exists()
            else {"plan_sha256": plan_id, "decisions": []}
        )
        if journal["plan_sha256"] != plan_id or len(journal["decisions"]) > len(candidates):
            raise ValueError("Search journal differs from registered plan")
        profile = None
        if not selection_path.exists():
            seed_everything(config.seed, config.device)
            profile_path = directory / "preflight.json"
            if not config.debug:
                if profile_path.exists():
                    profile = read_json(profile_path)
                    verify_profile(profile, root, manifest, config, methods=("erm", "cue"))
                else:
                    profile = profile_dataset(
                        root, manifest, config, profile_path, methods=("erm", "cue")
                    )
                    verify_profile(profile, root, manifest, config, methods=("erm", "cue"))
            erm_est = estimated_run_seconds(config, manifest, profile, "erm")
            cue_est = estimated_run_seconds(config, manifest, profile, "cue")
            if not baseline.exists() and not admit_candidate(
                time.time(),
                deadline,
                erm_est + cue_est,
                settings["reserve_minutes"],
                settings["time_safety_factor"],
            ):
                raise RuntimeError(
                    "Estimated time cannot fit ERM plus one full CUE trial before the deadline; no training or test started"
                )
            with DeadlineProgress(directory, deadline, config.epochs):
                summary = train(
                    config, manifest, baseline, root, selection_id=plan_id, resume=baseline.exists()
                )
                estimate = summary["elapsed_seconds"] * max(1, cue_est / max(erm_est, 1e-9))
                for index, candidate in enumerate(candidates):
                    folder = directory / "development" / f"cue{index}"
                    if index >= len(journal["decisions"]):
                        now = time.time()
                        admitted = admit_candidate(
                            now,
                            deadline,
                            estimate,
                            settings["reserve_minutes"],
                            settings["time_safety_factor"],
                        )
                        journal["decisions"].append(
                            {
                                "index": index,
                                "action": "run" if admitted else "skip_budget",
                                "decided_at": now,
                                "estimated_seconds": estimate,
                            }
                        )
                        atomic_json(journal_path, journal)
                    decision = journal["decisions"][index]
                    if decision["index"] != index:
                        raise ValueError("Candidate order changed")
                    write(
                        f"[search] candidate={index + 1}/{len(candidates)} action={decision['action']} tau={candidate.gate_tau} floor={candidate.joint_floor} estimated={duration(estimate)} deadline_remaining={duration(deadline - time.time())}"
                    )
                    if decision["action"] == "skip_budget":
                        continue
                    if decision["action"] != "run":
                        raise ValueError("Unknown search action")
                    result = train(
                        candidate,
                        manifest,
                        folder,
                        root,
                        phase="development",
                        selection_id=plan_id,
                        resume=folder.exists(),
                    )
                    estimate = max(estimate, result["elapsed_seconds"])
        rows = collect_candidates(
            root, directory, manifest, candidates, journal["decisions"], plan_id
        )
        if len(journal["decisions"]) != len(candidates) or not rows:
            raise RuntimeError(
                "No complete CUE candidate or search not closed; final test remains sealed"
            )
        chosen = winner(rows)
        selection = {
            "plan_sha256": plan_id,
            "search": journal,
            "candidates": rows,
            "selected": chosen,
            "selection_data": "validation_only_fixed_final_epoch",
        }
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ValueError("Frozen validation selection or search journal changed")
        atomic_json(selection_path, selection)
        selection_id = canonical_hash(selection)
        selected_config = candidates[chosen["index"]]
        promote_winner(
            root,
            directory / "development" / chosen["name"],
            selected,
            selected_config,
            manifest,
            selection_id,
        )
        specs = [
            (baseline, config, plan_id, "main"),
            (selected, selected_config, selection_id, "main"),
        ]
        contracts = [
            verify_run(folder, root, cfg, manifest["dataset_id"], sid)[0]
            for folder, cfg, sid, _ in specs
        ]
        if len({item["initial_shared_model_sha256"] for item in contracts}) != 1:
            raise ValueError("ERM and CUE initial weights differ")
        common = [
            {k: v for k, v in cfg.to_dict().items() if k not in CUE_FIELDS | {"method"}}
            for _, cfg, _, _ in specs
        ]
        if common[0] != common[1]:
            raise ValueError("ERM and CUE common settings differ")
        sealed_plan = {**plan, "selection_sha256": selection_id}
        seal = seal_study(directory, root, sealed_plan, specs, {"cifar100": manifest})
        manifest = load_manifest(path, root_override=data_root)
        if manifest["dataset_id"] != plan["dataset_id"]:
            raise ValueError("Dataset changed before final test")
        metrics = {}
        for spec in specs:
            evaluate_run(manifest, spec[0], root, seal)
            metrics[spec[1].method] = verified_metrics(root, manifest, spec, seal)
        output = {
            "status": "complete",
            "protocol": plan["protocol"],
            "split_audit": audit,
            "selection": selection,
            "test_metrics": metrics,
            "cue_minus_erm_pp": 100 * (metrics["cue"]["accuracy"] - metrics["erm"]["accuracy"]),
            "study_seal_sha256": canonical_hash(seal),
            "benchmark_evidence": not config.debug,
            "finished_KST": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "deadline_met": time.time() <= deadline,
        }
        atomic_json(directory / "comparison.json", output)
        lines = [
            "# One-seed PreAct ResNet-18 comparison",
            "",
            "Synthetic debug results, not benchmark evidence."
            if config.debug
            else "Train 45,000 / validation 5,000 / official test 10,000.",
            "",
            "Selection uses validation only. Test is evaluated after search closure.",
            "",
            "| Method | Seed | Final test accuracy |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {method} | {config.seed} | {100 * row['accuracy']:.2f}% |"
            for method, row in metrics.items()
        ]
        lines += [
            "",
            f"CUE - ERM: {output['cue_minus_erm_pp']:+.2f} pp.",
            f"Selected candidate: {chosen['name']}; {len(rows)} completed candidates.",
            "One seed; no variance or statistical significance claim. This split is not an exact reproduction of the published score.",
        ]
        (directory / "comparison.md").write_text("\n".join(lines) + "\n")
        atomic_json(
            directory / "progress.json",
            {
                "stage": "complete",
                "final_test_accessed": True,
                "deadline_met": output["deadline_met"],
            },
        )
        write(
            f"[complete] ERM test={100 * metrics['erm']['accuracy']:.2f}% CUE test={100 * metrics['cue']['accuracy']:.2f}% results={directory / 'comparison.md'}"
        )
        return output
