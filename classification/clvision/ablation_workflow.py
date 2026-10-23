"""Tune on validation, freeze one shared recipe, and evaluate matched ResNet ablations."""

import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

from filelock import FileLock

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json, run_lock

from .ablation_settings import AblationSettings
from .architecture import backbone_description, backbone_reference_note
from .cue_workflow import REFERENCE, make_cue_plan, verified_metrics
from .data import load_manifest
from .engine import evaluate_run, train, verify_run
from .identity import source_hash
from .profile import profile_dataset, verify_profile
from .progress import StudyProgress
from .workflow import check_plan, paired_summary, seal_study


def make_ablation_plan(root, manifests, settings):
    """Register candidate values and the conditional LR rule before opening training data."""
    plan = make_cue_plan(root, manifests, settings)
    plan.update(
        mode="matched_ablation",
        main_methods=list(settings.methods),
        hyperparameter_selection=settings.preview()["selection"],
        conditional_trial_field="Non-ERM trial learning_rate is replaced by the ERM validation winner",
        development_trials={
            name: [
                {"name": key, "config": config.to_dict()} for key, config in settings.trials(name)
            ]
            for name in settings.datasets
        },
    )
    return plan


def winner(rows):
    """Use final-epoch validation accuracy, then NLL, then registered order; never test scores."""
    if not rows or any(
        not all(math.isfinite(row["validation"][key]) for key in ("accuracy", "nll"))
        for row in rows
    ):
        raise ValueError("Selection requires complete finite validation scores")
    return min(
        enumerate(rows),
        key=lambda item: (
            -item[1]["validation"]["accuracy"],
            item[1]["validation"]["nll"],
            item[0],
        ),
    )[1]


def trial_rows(root, directory, manifest, trials, plan_id, execute=False):
    """Verify every declared development completion and reject final-test access in tuning."""
    rows = []
    for name, config in trials:
        folder = Path(directory) / "development" / config.dataset / name
        if (folder / "test_access.json").exists():
            raise ValueError("Development trial accessed final test")
        if execute:
            train(
                config,
                manifest,
                folder,
                root,
                phase="development",
                selection_id=plan_id,
                resume=folder.exists(),
            )
        contract, _, summary = verify_run(folder, root, config, manifest["dataset_id"], plan_id)
        if contract["phase"] != "development":
            raise ValueError("Invalid development phase")
        rows.append(
            {
                "name": name,
                "config": config.to_dict(),
                "validation": summary["final_validation"],
                "completion_sha256": file_hash(folder / "completion.json"),
            }
        )
    return rows


def select_dataset(root, directory, manifest, plan, settings, execute=False):
    """Choose the baseline LR first, then tune each regularizer at that same LR and seed."""
    dataset = manifest["dataset"]
    base = settings.recipe(dataset)
    rows = []
    if settings.tune:
        rows = trial_rows(
            root,
            directory,
            manifest,
            [(key, cfg) for key, cfg in settings.trials(dataset) if cfg.method == "erm"],
            canonical_hash(plan),
            execute,
        )
        base = replace(base, learning_rate=winner(rows)["config"]["learning_rate"])
        extra = trial_rows(
            root,
            directory,
            manifest,
            [
                (key, cfg)
                for key, cfg in settings.trials(dataset, base.learning_rate)
                if cfg.method != "erm"
            ],
            canonical_hash(plan),
            execute,
        )
        for method, field in (("cue", "gate_tau"), ("fixed_global_local", "fixed_lambda")):
            candidates = [row for row in extra if row["config"]["method"] == method]
            if candidates:
                base = replace(base, **{field: winner(candidates)["config"][field]})
        rows += extra
    payload = {
        "schema": 1,
        "plan_sha256": canonical_hash(plan),
        "dataset_id": manifest["dataset_id"],
        "selection_data": "internal_validation_only_final_epoch" if settings.tune else "fixed",
        "candidates": rows,
        "shared_config": base.to_dict(),
    }
    return {**payload, "selection_id": canonical_hash(payload)}


def ablation_specs(directory, settings, selections):
    """Use identical common settings for every paired arm, varying only the objective."""
    from .config import Config

    return [
        (
            Path(directory) / "main" / dataset / f"seed{seed}_{method}",
            Config.from_dict(
                {**selections[dataset]["shared_config"], "method": method, "seed": seed}
            ),
            selections[dataset]["selection_id"],
            "main",
        )
        for dataset in settings.datasets
        for seed in settings.seeds
        for method in settings.methods
    ]


def seal_ablation(directory, root, plan, specs, manifests):
    """Recompute validation choices and verify complete matched arms before any test access."""
    settings = AblationSettings.from_dict(plan["settings"])
    if plan != make_ablation_plan(root, manifests, settings):
        raise ValueError("Ablation scope or source changed")
    selections = {
        name: select_dataset(root, directory, manifests[name], plan, settings)
        for name in settings.datasets
    }
    for name, selection in selections.items():
        if read_json(Path(directory) / "development" / name / "selection.json") != selection:
            raise ValueError("Frozen validation selection changed")
    if specs != ablation_specs(directory, settings, selections):
        raise ValueError("Ablation omitted or substituted a registered arm/seed/recipe")
    shared = {}
    for folder, config, selection_id, _ in specs:
        contract, _, _ = verify_run(
            folder, root, config, manifests[config.dataset]["dataset_id"], selection_id
        )
        shared.setdefault((config.dataset, config.seed), set()).add(
            (
                contract["initial_shared_model_sha256"],
                contract["inference_parameters"],
            )
        )
    if any(len(values) != 1 for values in shared.values()):
        raise ValueError("Ablation arms have different initial weights or inference parameters")
    return seal_study(directory, root, plan, specs, manifests)


def ablation_report(directory, root, plan, specs, manifests, seal):
    """Report paired measured gains independently from descriptive published-score differences."""
    if seal != seal_ablation(directory, root, plan, specs, manifests):
        raise ValueError("Ablation study seal changed")
    references = read_json(Path(root) / REFERENCE)
    output = {
        "schema": 1,
        "mode": "matched_ablation",
        "source_sha256": source_hash(root),
        "benchmark_evidence": plan["benchmark_evidence"],
        "study_seal_sha256": canonical_hash(seal),
        "paired_comparison": True,
        "formal_significance_claim": False,
        "datasets": {},
        "published_reference": references,
    }
    lines = [
        "# Matched ResNet ablations",
        "",
        "Paired seed differences are descriptive; no formal significance claim.",
        "",
    ]
    if not plan["benchmark_evidence"]:
        lines += ["Synthetic debug results are not benchmark accuracy.", ""]
    for dataset in plan["natural_datasets"]:
        methods = {}
        config = next(spec[1] for spec in specs if spec[1].dataset == dataset)
        input_size = config.input_shape[0]
        backbone = backbone_description(config.activation)
        lines += [
            f"## {dataset}",
            "",
            f"Model input: {input_size} x {input_size} pixels for every arm and split.",
            f"Backbone: {backbone} (`{config.activation}`) for every arm.",
            "",
            "| Method | Measured accuracy (%) | Seed SD (pp) |",
            "|---|---:|---:|",
        ]
        for method in plan["main_methods"]:
            scores = {
                str(spec[1].seed): verified_metrics(root, manifests[dataset], spec, seal)
                for spec in specs
                if (spec[1].dataset, spec[1].method) == (dataset, method)
            }
            accuracies = [100 * value["accuracy"] for value in scores.values()]
            methods[method] = {
                "by_seed": scores,
                "mean_accuracy_percent": statistics.mean(accuracies),
                "seed_sd_pp": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
                "n_seeds": len(scores),
                "origin": "measured_here",
            }
            sd = methods[method]["seed_sd_pp"]
            lines.append(
                f"| {method} | {statistics.mean(accuracies):.2f} | {sd if sd is not None else 'unavailable'} |"
            )
        differences = {
            f"cue_minus_{method}_pp": paired_summary(
                [
                    100
                    * (
                        methods["cue"]["by_seed"][str(seed)]["accuracy"]
                        - methods[method]["by_seed"][str(seed)]["accuracy"]
                    )
                    for seed in plan["main_seeds"]
                ]
            )
            for method in plan["main_methods"]
            if method != "cue"
        }
        published = [row for row in references["scores"] if row["dataset"] == dataset]
        output["datasets"][dataset] = {
            "input_size": input_size,
            "activation": config.activation,
            "backbone": backbone,
            "methods": methods,
            "paired_differences": differences,
            "selection": read_json(Path(directory) / "development" / dataset / "selection.json"),
            "published_references": published,
            "published_comparison_is_matched": False,
        }
        lines += [
            "",
            *[
                f"- {key}: {value['mean']:+.3f} pp; by seed: {value['by_seed']}"
                for key, value in differences.items()
            ],
            "",
        ]
        lines += [
            "Published references below use a different protocol and were not reproduced here.",
            "",
            "| Method | Published accuracy (%) |",
            "|---|---:|",
        ]
        lines += [f"| {row['method']} | {row['mean_accuracy_percent']:.2f} |" for row in published]
        lines += [
            "",
            backbone_reference_note(config.activation),
            "Split, batch/LR/warmup, and seed count must also match for a protocol reproduction; "
            f"our input is {input_size}px and the published input is 32px. Matching resolution alone does not match the full protocol.",
            "",
        ]
    lines += [f"Source: [Table 2]({references['source']['url']}).", ""]
    lines += ["> " + quote for quote in references["source"]["excerpts"]]
    atomic_json(Path(directory) / "comparison.json", output)
    (Path(directory) / "comparison.md").write_text("\n".join(lines) + "\n")
    return output


def run_ablation(root, data_root, directory, settings):
    """Run only the registered tuning and ablation budget, sealing all training before tests."""
    root, data_root, directory = Path(root), Path(data_root), Path(directory)
    started = time.monotonic()
    lock = root / ".cache/classification_device.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0), run_lock(directory):
        manifests = {
            name: load_manifest(data_root / "manifests" / f"{name}.json", root_override=data_root)
            for name in settings.datasets
        }
        plan = make_ablation_plan(root, manifests, settings)
        check_plan(directory, plan)
        atomic_json(directory / "plan.json", plan)
        profiles = {}
        if not settings.debug:
            for name in settings.datasets:
                config = settings.recipe(name)
                if settings.tune:
                    config = replace(config, learning_rate=max(settings.lr_candidates))
                path = directory / "preflight" / f"{name}.json"
                profiles[name] = profile_dataset(
                    root, manifests[name], config, path, methods=settings.methods
                )
                verify_profile(
                    profiles[name], root, manifests[name], config, methods=settings.methods
                )
        with StudyProgress(directory, plan, manifests, profiles, started):
            selections = {}
            for name in settings.datasets:
                selection = select_dataset(
                    root, directory, manifests[name], plan, settings, execute=True
                )
                path = directory / "development" / name / "selection.json"
                if path.exists() and read_json(path) != selection:
                    raise ValueError("Frozen validation selection changed")
                atomic_json(path, selection)
                selections[name] = selection
                print(
                    f"[selection] {name}: LR={selection['shared_config']['learning_rate']} tau={selection['shared_config']['gate_tau']} internal_validation_only={settings.tune}",
                    flush=True,
                )
            specs = ablation_specs(directory, settings, selections)
            for folder, config, selection_id, phase in specs:
                train(
                    config,
                    manifests[config.dataset],
                    folder,
                    root,
                    phase=phase,
                    selection_id=selection_id,
                    resume=folder.exists(),
                )
            seal = seal_ablation(directory, root, plan, specs, manifests)
            for folder, config, _, _ in specs:
                evaluate_run(manifests[config.dataset], folder, root, seal)
            return ablation_report(directory, root, plan, specs, manifests, seal)
