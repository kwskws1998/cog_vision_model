"""Frozen development choices, complete matched matrices, and auditable study-level testing."""

import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json, run_lock
from cue_routing.metrics import classification_metrics

from .architecture import backbone_description
from .config import CORRUPTIONS, MAIN_METHODS, MECHANISM_METHODS
from .data import ImageDataset, load_manifest
from .engine import evaluate_corruptions, evaluate_run, train, verify_run
from .identity import source_hash
from .profile import profile_dataset, verify_profile
from .progress import StudyProgress
from .settings import CANDIDATE_FIELDS, resolve_settings


def base_recipe(dataset, settings):
    """Resolve the same validated recipe for preflight, development, and study training."""
    return resolve_settings(settings).recipe(dataset)


def candidate_configs(base, settings=None):
    """Allocate an explicitly matched validation-only budget to every tunable intervention."""
    settings = resolve_settings(settings)
    if base.dataset not in settings.development_candidates:
        raise ValueError("Development candidates are defined only for natural-image datasets")
    result = []
    for method, key in CANDIDATE_FIELDS.items():
        values = settings.development_candidates[base.dataset][method]
        for index, value in enumerate(values):
            result.append(
                (
                    f"{method}_candidate{index}",
                    replace(
                        base, method=method, seed=settings.development_seed, **{key: value}
                    ).validate(),
                )
            )
    return result


def choose_candidates(rows):
    """Select final-epoch validation accuracy, then lower NLL, then registered candidate order."""
    counts = [sum(row["config"]["method"] == method for row in rows) for method in CANDIDATE_FIELDS]
    if not counts[0] or len(set(counts)) != 1 or sum(counts) != len(rows):
        raise ValueError("Each tunable method requires the same positive count of completed trials")
    selected = {}
    for method in ("cue", "fixed_global_local", "zipfls_dense2"):
        options = [
            (index, row) for index, row in enumerate(rows) if row["config"]["method"] == method
        ]
        if any(
            not all(math.isfinite(row["validation"][key]) for key in ("accuracy", "nll"))
            for _, row in options
        ):
            raise ValueError("Nonfinite development score")
        _, winner = min(
            options,
            key=lambda item: (
                -item[1]["validation"]["accuracy"],
                item[1]["validation"]["nll"],
                item[0],
            ),
        )
        selected[method] = winner["config"]
    return selected


def develop(root, manifest, directory, base, settings=None):
    """Freeze every registered development score and never construct a final-test loader."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, config in candidate_configs(base, settings):
        run_dir = directory / name
        train(config, manifest, run_dir, root, phase="development", resume=run_dir.exists())
        contract, _, summary = verify_run(run_dir, root, config, manifest["dataset_id"], "")
        if contract["phase"] != "development" or (run_dir / "test_access.json").exists():
            raise ValueError("Development artifacts have invalid phase or final-test access")
        rows.append(
            {
                "name": name,
                "config": config.to_dict(),
                "validation": summary["final_validation"],
                "completion_sha256": file_hash(run_dir / "completion.json"),
            }
        )
    payload = {
        "schema": 1,
        "source_sha256": source_hash(root),
        "dataset_id": manifest["dataset_id"],
        "candidates": rows,
        "selected": choose_candidates(rows),
        "selection_data": "internal_validation_only_final_epoch",
    }
    selection = {**payload, "selection_id": canonical_hash(payload)}
    path = directory / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("Frozen development selection changed; use a separately registered study")
    atomic_json(path, selection)
    return selection


def selected_config(base, selection, method, seed):
    """Transfer only the registered intervention settings into a paired main run."""
    if selection["selection_data"] != "internal_validation_only_final_epoch":
        raise ValueError("Invalid development provenance")
    config = base
    keys = {
        "cue": ("gate_tau",),
        "fixed_global_local": ("fixed_lambda",),
        "zipfls_dense2": ("zipf_lambda",),
    }
    if method in keys:
        config = replace(
            config, **{key: selection["selected"][method][key] for key in keys[method]}
        )
    return replace(config, method=method, seed=seed).validate()


def main_specs(root, selection, base, seeds):
    """Enumerate every arm/seed in a stable order before final results exist."""
    return [
        (
            Path(root) / base.dataset / f"seed{seed}_{method}",
            selected_config(base, selection, method, seed),
            selection["selection_id"],
            "main",
        )
        for seed in seeds
        for method in MAIN_METHODS
    ]


def mechanism_specs(root, selection, base, seeds):
    """Reuse CIFAR validation choices without fitting the independent/reversed final regimes."""
    chosen_tau = selection["selected"]["cue"]["gate_tau"]
    fixed = selection["selected"]["fixed_global_local"]["fixed_lambda"]
    return [
        (
            Path(root) / "dominoes" / f"seed{seed}_{method}",
            replace(
                base, method=method, seed=seed, gate_tau=chosen_tau, fixed_lambda=fixed
            ).validate(),
            selection["selection_id"],
            "mechanism",
        )
        for seed in seeds
        for method in MECHANISM_METHODS
    ]


def make_plan(manifests, root, settings):
    """Resolve the full study recipe and every development trial before starting any training."""
    settings = resolve_settings(settings)
    bases = {
        dataset: settings.recipe(dataset) for dataset in ("cifar100", "tiny_imagenet", "dominoes")
    }
    plan = {
        "schema": 2,
        "source_sha256": source_hash(root),
        "settings": settings.to_dict(),
        "resolved_recipes": {dataset: config.to_dict() for dataset, config in bases.items()},
        "development_trials": {
            dataset: [
                {"name": name, "config": config.to_dict()}
                for name, config in candidate_configs(bases[dataset], settings)
            ]
            for dataset in ("cifar100", "tiny_imagenet")
        },
        "dataset_ids": {name: manifest["dataset_id"] for name, manifest in manifests.items()},
        "main_methods": list(MAIN_METHODS),
        "mechanism_methods": list(MECHANISM_METHODS),
        "main_seeds": list(settings.main_seeds),
        "mechanism_seeds": list(settings.mechanism_seeds),
        "natural_datasets": ["cifar100", "tiny_imagenet"],
        "final_selection": "fixed_final_epoch",
        "benchmark_evidence": not settings.debug,
    }
    return plan


def check_plan(study_dir, plan):
    """Refuse changed study settings or source before preflight or resumed training can run."""
    path = Path(study_dir) / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError(
            "Study scope or source changed; resume cannot silently change the experiment"
        )


def freeze_plan(study_dir, manifests, root, settings):
    """Write complete settings, resolved recipes, and dataset/arm/seed scope before training."""
    plan = make_plan(manifests, root, settings)
    check_plan(study_dir, plan)
    path = Path(study_dir) / "plan.json"
    atomic_json(path, plan)
    return plan


def seal_study(study_dir, root, plan, specs, manifests):
    """Validate every planned main/mechanism run before granting any final-test access."""
    expected = len(plan["main_seeds"]) * len(plan.get("main_methods", MAIN_METHODS)) * len(
        plan.get("natural_datasets", ("cifar100", "tiny_imagenet"))
    ) + len(plan["mechanism_seeds"]) * len(plan.get("mechanism_methods", MECHANISM_METHODS))
    if len(specs) != expected or len({str(row[0].resolve()) for row in specs}) != expected:
        raise ValueError("Study matrix omitted or duplicated a prespecified run")
    runs = {}
    for run_dir, config, selection_id, phase in specs:
        contract, _, _ = verify_run(
            run_dir, root, config, manifests[config.dataset]["dataset_id"], selection_id
        )
        if contract["phase"] != phase:
            raise ValueError("Study phase differs from its registered scope")
        runs[str(run_dir.resolve())] = file_hash(run_dir / "completion.json")
    value = {
        "schema": 1,
        "purpose": "final_evaluation_after_all_prespecified_training",
        "source_sha256": source_hash(root),
        "plan_sha256": canonical_hash(plan),
        "runs": runs,
        "benchmark_evidence": plan["benchmark_evidence"],
    }
    path = Path(study_dir) / "study_seal.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Whole-study training seal changed after final evaluation")
    atomic_json(path, value)
    return value


def paired_summary(differences):
    """Describe paired differences without treating a pilot with few seeds as confirmation."""
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(17011)
    resampled = generator.choice(values, size=(10000, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "by_seed": values.tolist(),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "descriptive_bootstrap_95_interval": np.quantile(resampled, [0.025, 0.975]).tolist(),
        "n_seeds": len(values),
        "formal_significance_claim": False,
    }


def corruption_summary(artifact, contract_hash, seal_hash, dataset_id, count):
    """Require the complete declared corruption grid and derive aggregates from its cells."""
    identity = {
        "contract_hash": contract_hash,
        "study_seal_sha256": seal_hash,
        "dataset_id": dataset_id,
    }
    expected = {f"{name}/{severity}" for name in CORRUPTIONS for severity in range(1, 6)}
    cells = artifact.get("cells", {})
    if any(artifact.get(key) != value for key, value in identity.items()) or set(cells) != expected:
        raise ValueError("Corruption identity or complete 75-condition grid differs")
    if any(
        value.get("n") != count
        or not isinstance(value.get("accuracy"), (int, float))
        or not math.isfinite(value["accuracy"])
        or not 0 <= value["accuracy"] <= 1
        for value in cells.values()
    ):
        raise ValueError("Corruption sample counts or accuracy values are invalid")
    mean = statistics.mean(value["accuracy"] for value in cells.values())
    if not math.isclose(
        mean, artifact.get("mean_corruption_accuracy", float("nan")), abs_tol=1e-12
    ):
        raise ValueError("Corruption aggregate differs from its saved cells")
    return {
        "mean_accuracy": mean,
        "mean_error": 1 - mean,
        "by_corruption": {
            name: statistics.mean(
                cells[f"{name}/{severity}"]["accuracy"] for severity in range(1, 6)
            )
            for name in CORRUPTIONS
        },
        "by_severity": {
            str(severity): statistics.mean(
                cells[f"{name}/{severity}"]["accuracy"] for name in CORRUPTIONS
            )
            for severity in range(1, 6)
        },
        "cells": cells,
    }


def report(study_dir, root, specs, manifests, seal):
    """Reject unmatched or stale predictions and include all methods and all declared datasets."""
    study_dir = Path(study_dir)
    output = {
        "schema": 1,
        "source_sha256": source_hash(root),
        "study_seal_sha256": canonical_hash(seal),
        "benchmark_evidence": seal["benchmark_evidence"],
        "datasets": {},
        "uncertainty": "Pilot with few paired seeds; bootstrap intervals are descriptive and unstable; no formal significance claim",
    }
    tables = []
    for dataset in ("cifar100", "tiny_imagenet", "dominoes"):
        selected = [row for row in specs if row[1].dataset == dataset]
        activations = {row[1].activation for row in selected}
        if len(activations) != 1:
            raise ValueError("A dataset comparison requires one matched backbone activation")
        activation = next(iter(activations))
        backbone = backbone_description(activation)
        rows, shared, contracts = [], {}, {}
        expected_data = ImageDataset(manifests[dataset], "test")
        expected_ids = [expected_data.sample_id(index) for index in range(len(expected_data))]
        expected_targets = torch.tensor(
            [expected_data.sample_target(index) for index in range(len(expected_data))],
            dtype=torch.long,
        )
        for run_dir, config, selection_id, _ in selected:
            contract, _, _ = verify_run(
                run_dir, root, config, manifests[dataset]["dataset_id"], selection_id
            )
            contracts[(config.seed, config.method)] = canonical_hash(contract)
            value = read_json(run_dir / "test_metrics.json")
            if (
                value["contract_hash"] != canonical_hash(contract)
                or value["study_seal_sha256"] != canonical_hash(seal)
                or value["prediction_sha256"] != file_hash(run_dir / "test_predictions.pt")
            ):
                raise ValueError("Evaluation artifacts do not match the sealed run")
            predictions = torch.load(
                run_dir / "test_predictions.pt", map_location="cpu", weights_only=True
            )
            if (
                predictions["sample_ids"] != expected_ids
                or not torch.equal(predictions["targets"], expected_targets)
                or classification_metrics(predictions["logits"], predictions["targets"])
                != value["metrics"]
            ):
                raise ValueError(
                    "Saved predictions do not support the reported test metrics or order"
                )
            signature = canonical_hash(
                {"ids": predictions["sample_ids"], "targets": predictions["targets"].tolist()}
            )
            shared.setdefault(config.seed, []).append(
                (
                    contract["initial_shared_model_sha256"],
                    signature,
                    contract["inference_parameters"],
                )
            )
            rows.append((config, value))
        if any(len(set(group)) != 1 for group in shared.values()):
            raise ValueError(
                "Methods have unmatched initial weights, test identities, or inference parameters"
            )
        methods = MECHANISM_METHODS if dataset == "dominoes" else MAIN_METHODS
        summary = {}
        for method in methods:
            values = [value for config, value in rows if config.method == method]
            accuracies = [value["metrics"]["accuracy"] for value in values]
            summary[method] = {
                "mean_accuracy": statistics.mean(accuracies),
                "seed_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
                "by_seed": {
                    str(value["seed"]): {
                        key: item for key, item in value["metrics"].items() if key != "confusion"
                    }
                    for value in values
                },
            }
            if dataset == "dominoes":
                summary[method]["cue_conflict_by_seed"] = {
                    str(value["seed"]): value["cue_conflict"] for value in values
                }
            if dataset == "cifar100" and "cifar100_c" in manifests:
                corruption = {}
                for run_dir, config, _, _ in selected:
                    if config.method == method:
                        if not (run_dir / "corruption_metrics.json").is_file():
                            raise ValueError("Corruption matrix is incomplete")
                        artifact = read_json(run_dir / "corruption_metrics.json")
                        corruption[str(config.seed)] = corruption_summary(
                            artifact,
                            contracts[(config.seed, method)],
                            canonical_hash(seal),
                            manifests["cifar100_c"]["dataset_id"],
                            len(expected_data),
                        )
                means = [item["mean_accuracy"] for item in corruption.values()]
                summary[method]["corruptions"] = {
                    "mean_accuracy": statistics.mean(means),
                    "mean_error": 1 - statistics.mean(means),
                    "seed_sd": statistics.stdev(means) if len(means) > 1 else None,
                    "by_seed": corruption,
                    "by_corruption": {
                        name: statistics.mean(
                            item["by_corruption"][name] for item in corruption.values()
                        )
                        for name in CORRUPTIONS
                    },
                    "by_severity": {
                        str(severity): statistics.mean(
                            item["by_severity"][str(severity)] for item in corruption.values()
                        )
                        for severity in range(1, 6)
                    },
                }
        differences = {}
        cue_values = summary["cue"]["by_seed"]
        for method in methods:
            if method != "cue":
                differences[f"cue_minus_{method}"] = paired_summary(
                    [
                        cue_values[seed]["accuracy"] - summary[method]["by_seed"][seed]["accuracy"]
                        for seed in sorted(cue_values, key=int)
                    ]
                )
        output["datasets"][dataset] = {
            "activation": activation,
            "backbone": backbone,
            "methods": summary,
            "paired_differences": differences,
            "ranking": sorted(methods, key=lambda method: -summary[method]["mean_accuracy"]),
        }
        if dataset == "cifar100" and "cifar100_c" in manifests:
            output["datasets"][dataset]["paired_corruption_differences"] = {
                f"cue_minus_{method}": paired_summary(
                    [
                        summary["cue"]["corruptions"]["by_seed"][seed]["mean_accuracy"]
                        - summary[method]["corruptions"]["by_seed"][seed]["mean_accuracy"]
                        for seed in sorted(cue_values, key=int)
                    ]
                )
                for method in methods
                if method != "cue"
            }
        tables.extend(
            [
                f"## {dataset}",
                "",
                f"Backbone: {backbone} (`{activation}`).",
                "",
                "| Method | Accuracy (%) | Seed SD (pp) |",
                "|---|---:|---:|",
            ]
        )
        for method in output["datasets"][dataset]["ranking"]:
            item = summary[method]
            sd = f"{100 * item['seed_sd']:.2f}" if item["seed_sd"] is not None else "unavailable"
            tables.append(f"| {method} | {100 * item['mean_accuracy']:.2f} | {sd} |")
        tables.append("")
        if dataset == "cifar100" and "cifar100_c" in manifests:
            tables.extend(
                [
                    "### CIFAR-100-C",
                    "",
                    "| Method | Mean accuracy (%) | Seed SD (pp) |",
                    "|---|---:|---:|",
                ]
            )
            for method in sorted(
                methods, key=lambda name: -summary[name]["corruptions"]["mean_accuracy"]
            ):
                item = summary[method]["corruptions"]
                sd = (
                    f"{100 * item['seed_sd']:.2f}" if item["seed_sd"] is not None else "unavailable"
                )
                tables.append(f"| {method} | {100 * item['mean_accuracy']:.2f} | {sd} |")
            tables.append("")
    atomic_json(study_dir / "comparison.json", output)
    (study_dir / "comparison.md").write_text(
        "# Matched classification results\n\n"
        + "\n".join(tables)
        + "\nBootstrap intervals are descriptive; synthetic/debug runs are not benchmark evidence.\n"
    )
    return output


def run_study(root, data_root, study_dir, debug=None, workers=None, settings=None):
    """Exclude concurrent classification jobs before either CUDA preflight or study training."""
    lock_path = Path(root) / ".cache/classification_device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=0):
        return _run_study(root, data_root, study_dir, debug, workers, settings)


def _run_study(root, data_root, study_dir, debug=None, workers=None, settings=None):
    """Run development, freeze choices, train all matrices, then evaluate clean and corrupted tests."""
    root, data_root, study_dir = Path(root), Path(data_root), Path(study_dir)
    started = time.monotonic()
    settings = resolve_settings(settings, debug=debug, workers=workers)
    debug = settings.debug
    manifests = {}
    for name in ("cifar100", "tiny_imagenet", "dominoes"):
        print(f"[prepare] Verifying {name} data", flush=True)
        manifests[name] = load_manifest(
            data_root / "manifests" / f"{name}.json", root_override=data_root
        )
    if not debug:
        print("[prepare] Verifying cifar100_c data", flush=True)
        manifests["cifar100_c"] = load_manifest(
            data_root / "manifests" / "cifar100_c.json", root_override=data_root
        )
    intended_plan = make_plan(manifests, root, settings)
    check_plan(study_dir, intended_plan)
    profiles = {}
    if not debug:
        for dataset in ("cifar100", "tiny_imagenet", "dominoes"):
            base = base_recipe(dataset, settings)
            path = root / "classification" / "verification" / f"{dataset}_cuda.json"
            value = profile_dataset(root, manifests[dataset], base, path)
            verify_profile(value, root, manifests[dataset], base)
            profiles[dataset] = value
    with run_lock(study_dir), StudyProgress(study_dir, intended_plan, manifests, profiles, started):
        plan = freeze_plan(study_dir, manifests, root, settings)
        specs = []
        cifar_base = base_recipe("cifar100", settings)
        cifar_selection = develop(
            root,
            manifests["cifar100"],
            study_dir / "development" / "cifar100",
            cifar_base,
            settings,
        )
        mechanism = mechanism_specs(
            study_dir / "main",
            cifar_selection,
            base_recipe("dominoes", settings),
            plan["mechanism_seeds"],
        )
        for run_dir, config, selection_id, phase in mechanism:
            train(
                config,
                manifests[config.dataset],
                run_dir,
                root,
                phase,
                selection_id,
                resume=run_dir.exists(),
            )
        specs.extend(mechanism)
        for dataset in ("cifar100", "tiny_imagenet"):
            base = base_recipe(dataset, settings)
            selection = (
                cifar_selection
                if dataset == "cifar100"
                else develop(
                    root, manifests[dataset], study_dir / "development" / dataset, base, settings
                )
            )
            matrix = main_specs(study_dir / "main", selection, base, plan["main_seeds"])
            for index, (run_dir, config, selection_id, phase) in enumerate(matrix, start=1):
                print(
                    f"[{index}/{len(matrix)}] {dataset} {config.method} seed={config.seed}",
                    flush=True,
                )
                train(
                    config,
                    manifests[dataset],
                    run_dir,
                    root,
                    phase,
                    selection_id,
                    resume=run_dir.exists(),
                )
            specs.extend(matrix)
        seal = seal_study(study_dir, root, plan, specs, manifests)
        for run_dir, config, _, _ in specs:
            evaluate_run(manifests[config.dataset], run_dir, root, seal)
        if not debug:
            for run_dir, config, _, _ in specs:
                if config.dataset == "cifar100":
                    evaluate_corruptions(
                        manifests["cifar100_c"], manifests["cifar100"], run_dir, root, seal
                    )
        return report(study_dir, root, specs, manifests, seal)
