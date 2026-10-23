"""Train one fixed method, seal the declared seeds, and report independently verified results."""

import statistics
import time
from pathlib import Path

import torch
from filelock import FileLock

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json, run_lock
from cue_routing.metrics import classification_metrics

from .architecture import backbone_description, backbone_reference_note
from .config import Config
from .data import ImageDataset, load_manifest
from .engine import evaluate_run, train, verify_run
from .identity import source_hash
from .profile import profile_dataset, verify_profile
from .progress import StudyProgress
from .workflow import check_plan, seal_study

REFERENCE = "classification/references/published_classification.json"


def make_cue_plan(root, manifests, settings):
    """Register one fixed method and only the requested natural-image jobs before training."""
    method = settings.recipe(settings.datasets[0]).method
    return {
        "schema": 1,
        "mode": f"{method}_only",
        "source_sha256": source_hash(root),
        "settings": settings.to_dict(),
        "resolved_recipes": {name: settings.recipe(name).to_dict() for name in settings.datasets},
        "dataset_ids": {name: manifests[name]["dataset_id"] for name in settings.datasets},
        "development_trials": {},
        "main_methods": [method],
        "main_seeds": list(settings.seeds),
        "mechanism_methods": [],
        "mechanism_seeds": [],
        "natural_datasets": list(settings.datasets),
        "final_selection": "fixed_final_epoch",
        "hyperparameter_selection": "fixed_before_training",
        "evaluate_corruptions": False,
        "benchmark_evidence": not settings.debug,
        "published_reference_sha256": file_hash(Path(root) / REFERENCE),
    }


def cue_specs(directory, plan):
    """Derive the exact run paths and identities from the frozen plan."""
    selection_id = canonical_hash(plan)
    return [
        (
            Path(directory)
            / "main"
            / dataset
            / f"seed{seed}_{plan['resolved_recipes'][dataset]['method']}",
            Config.from_dict({**plan["resolved_recipes"][dataset], "seed": seed}),
            selection_id,
            "main",
        )
        for dataset in plan["natural_datasets"]
        for seed in plan["main_seeds"]
    ]


def seal_cue_study(directory, root, plan, specs, manifests):
    """Reject a substituted method, seed, path, or recipe before opening any final tests."""
    methods = {value["method"] for value in plan["resolved_recipes"].values()}
    if (
        len(methods) != 1
        or not methods <= {"erm", "cue"}
        or plan["main_methods"] != sorted(methods)
        or plan["mode"] != f"{next(iter(methods))}_only"
        or plan["source_sha256"] != source_hash(root)
        or specs != cue_specs(directory, plan)
    ):
        raise ValueError("CUE run scope or source differs from the frozen plan")
    return seal_study(directory, root, plan, specs, manifests)


def verified_metrics(root, manifest, spec, seal):
    """Recompute metrics from identity-checked predictions, including on a completed-run restart."""
    run_dir, config, selection_id, _ = spec
    contract, _, _ = verify_run(run_dir, root, config, manifest["dataset_id"], selection_id)
    value = read_json(run_dir / "test_metrics.json")
    contract_hash = canonical_hash(contract)
    if (
        value["contract_hash"] != contract_hash
        or value["study_seal_sha256"] != canonical_hash(seal)
        or value["prediction_sha256"] != file_hash(run_dir / "test_predictions.pt")
        or (value["dataset"], value["method"], value["seed"])
        != (config.dataset, config.method, config.seed)
    ):
        raise ValueError("Evaluation artifacts do not match the sealed CUE run")
    predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=True)
    expected = ImageDataset(manifest, "test", tiny_imagenet_size=config.tiny_imagenet_size)
    labels = torch.tensor([expected.sample_target(index) for index in range(len(expected))])
    if (
        predictions["contract_hash"] != contract_hash
        or predictions["sample_ids"]
        != [expected.sample_id(index) for index in range(len(expected))]
        or not torch.equal(predictions["targets"], labels)
        or tuple(predictions["logits"].shape) != (len(expected), len(manifest["classes"]))
        or classification_metrics(predictions["logits"], predictions["targets"]) != value["metrics"]
    ):
        raise ValueError("Saved predictions do not support CUE metrics or sample order")
    return value["metrics"]


def cue_report(directory, root, plan, manifests, specs, seal):
    """Keep local measurements and published reference scores explicitly separate in the comparison."""
    if seal != seal_cue_study(directory, root, plan, specs, manifests):
        raise ValueError("CUE study seal differs")
    references = read_json(Path(root) / REFERENCE)
    method = plan["main_methods"][0]
    if file_hash(Path(root) / REFERENCE) != plan["published_reference_sha256"]:
        raise ValueError("Published reference file changed after registration")
    output = {
        "schema": 1,
        "mode": plan["mode"],
        "source_sha256": plan["source_sha256"],
        "study_seal_sha256": canonical_hash(seal),
        "benchmark_evidence": plan["benchmark_evidence"],
        "published_reference": references,
        "paired_comparison": False,
        "formal_significance_claim": False,
        "datasets": {},
    }
    lines = [
        f"# {method.upper()} measurements and published references",
        "",
        "Cross-protocol descriptive comparison; published scores were not reproduced locally.",
        "No paired significance or method-superiority claim follows from this table.",
        "",
    ]
    if not plan["benchmark_evidence"]:
        lines += ["Synthetic debug run: these measurements are not benchmark accuracy.", ""]
    for dataset in plan["natural_datasets"]:
        selected = [spec for spec in specs if spec[1].dataset == dataset]
        scores = {
            str(spec[1].seed): verified_metrics(root, manifests[dataset], spec, seal)
            for spec in selected
        }
        accuracies = [100 * value["accuracy"] for value in scores.values()]
        measured = {
            "origin": "measured_here",
            "mean_accuracy_percent": statistics.mean(accuracies),
            "seed_sd_pp": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
            "n_seeds": len(scores),
            "by_seed": scores,
            "config": plan["resolved_recipes"][dataset],
            "split_counts": {
                name: len(rows) for name, rows in manifests[dataset]["splits"].items()
            },
            "activation": selected[0][1].activation,
            "backbone": backbone_description(selected[0][1].activation),
            "input_size": selected[0][1].input_shape[0],
        }
        published = [row for row in references["scores"] if row["dataset"] == dataset]
        differences = (
            [
                {
                    "method": row["method"],
                    f"{method}_minus_reference_pp": measured["mean_accuracy_percent"]
                    - row["mean_accuracy_percent"],
                    "matched_protocol": False,
                }
                for row in published
            ]
            if plan["benchmark_evidence"]
            else []
        )
        differences_note = [
            backbone_reference_note(measured["activation"]),
            "Our training holds out an internal validation split; the table does not establish identical split indices.",
            f"Our input size is {measured['input_size']}px; the paper uses 32px for both datasets.",
            f"Our batch/LR/warmup: {measured['config']['batch_size']}/"
            f"{measured['config']['learning_rate']}/{measured['config']['warmup_epochs']} epochs; "
            "the paper reports batch 128 and initial LR 0.1, without specifying warmup.",
            f"Our seed count is {len(scores)}; published mean/SD uses five seeds.",
        ]
        output["datasets"][dataset] = {
            f"{method}_measured": measured,
            "published_references": published,
            "descriptive_differences": differences,
            "protocol_notes": differences_note,
        }
        sd = (
            f"{measured['seed_sd_pp']:.2f}" if measured["seed_sd_pp"] is not None else "unavailable"
        )
        lines += [
            f"## {dataset}",
            "",
            f"Backbone: {measured['backbone']} (`{measured['activation']}`).",
            "",
            "| Method | Origin | Accuracy (%) | Seed SD (pp) | Seeds |",
            "|---|---|---:|---:|---:|",
            f"| {method.upper()} | measured here | {measured['mean_accuracy_percent']:.2f} | {sd} | {len(scores)} |",
        ]
        for row in published:
            lines.append(
                f"| {row['method']} | published, Table 2 | {row['mean_accuracy_percent']:.2f} | "
                f"{row['seed_sd_pp']:.2f} | {row['n_seeds']} |"
            )
        lines += ["", *["- " + note for note in differences_note], ""]
    source = references["source"]
    lines += [f"Source: [{source['title']}, Table 2 and section 4.1]({source['url']}).", ""]
    for quote in source["excerpts"]:
        lines += ["> " + quote, ""]
    atomic_json(Path(directory) / "comparison.json", output)
    (Path(directory) / "comparison.md").write_text("\n".join(lines))
    return output


def run_cue_study(root, data_root, directory, settings):
    """Profile only the chosen method, train all registered seeds, then seal, evaluate, and report."""
    root, data_root, directory = Path(root), Path(data_root), Path(directory)
    started = time.monotonic()
    lock_path = root / ".cache/classification_device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=0), run_lock(directory):
        manifests = {}
        for name in settings.datasets:
            print(f"[prepare] Verifying {name} data", flush=True)
            manifests[name] = load_manifest(
                data_root / "manifests" / f"{name}.json", root_override=data_root
            )
        plan = make_cue_plan(root, manifests, settings)
        check_plan(directory, plan)
        atomic_json(directory / "plan.json", plan)
        profiles = {}
        if not settings.debug:
            for name in settings.datasets:
                config = settings.recipe(name)
                methods = (config.method,)
                path = directory / "preflight" / f"{name}.json"
                profiles[name] = profile_dataset(
                    root, manifests[name], config, path, methods=methods
                )
                verify_profile(profiles[name], root, manifests[name], config, methods=methods)
        specs = cue_specs(directory, plan)
        with StudyProgress(directory, plan, manifests, profiles, started):
            for run_dir, config, selection_id, phase in specs:
                train(
                    config,
                    manifests[config.dataset],
                    run_dir,
                    root,
                    phase=phase,
                    selection_id=selection_id,
                    resume=run_dir.exists(),
                )
            seal = seal_cue_study(directory, root, plan, specs, manifests)
            for run_dir, config, _, _ in specs:
                evaluate_run(manifests[config.dataset], run_dir, root, seal)
            return cue_report(directory, root, plan, manifests, specs, seal)
