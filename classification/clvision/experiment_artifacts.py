"""Validate split isolation and preserve immutable validation-selected checkpoints."""

import torch

from cue_routing.io import atomic_json, canonical_hash, file_hash, read_json, run_lock, state_hash

from .engine import seal_run, verify_run


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
    return {
        "counts": counts,
        "source_split_policy": manifest.get("split_policy"),
        "pairwise_disjoint": True,
        "identity_hashes": {key: canonical_hash(sorted(items)) for key, items in ids.items()},
        "test_policy": "deferred until search closure; no test-driven tuning",
    }


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
