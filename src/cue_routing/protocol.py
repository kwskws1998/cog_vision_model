"""Source-validation selection and auditable, sealed target evaluation."""

from pathlib import Path

from .io import atomic_json, canonical_hash, file_hash, read_json


def is_better(candidate, incumbent):
    """Select higher source macro accuracy, then lower NLL, then the earlier step."""
    if incumbent is None:
        return True
    return (-candidate["macro_accuracy"], candidate["macro_nll"], candidate["step"]) < (
        -incumbent["macro_accuracy"],
        incumbent["macro_nll"],
        incumbent["step"],
    )


def seal_selection(run_dir, contract, selection, completed_steps):
    """Bind the selected weights to a completed, immutable source-only training run."""
    run_dir = Path(run_dir)
    value = {
        "schema": 1,
        "contract_hash": canonical_hash(contract),
        "best_checkpoint_sha256": file_hash(run_dir / "best.pt"),
        "selection": selection,
        "completed_steps": completed_steps,
        "target": contract["config"]["target"],
        "selection_data": "source_crossval_only",
    }
    atomic_json(run_dir / "selection.json", value)
    return value


def verify_selection(run_dir, contract):
    """Reject incomplete, altered, or mismatched artifacts before loading a target."""
    run_dir = Path(run_dir)
    seal = read_json(run_dir / "selection.json")
    if seal["contract_hash"] != canonical_hash(contract) or seal[
        "best_checkpoint_sha256"
    ] != file_hash(run_dir / "best.pt"):
        raise ValueError("Selection contract or selected checkpoint changed")
    if (
        seal["completed_steps"] != contract["config"]["steps"]
        or seal["selection_data"] != "source_crossval_only"
        or seal["target"] != contract["config"]["target"]
    ):
        raise ValueError("Incomplete training or invalid target-selection provenance")
    return seal


def record_target_access(run_dir, seal):
    """Journal explicit post-selection target access, including repeated evaluation."""
    path = Path(run_dir) / "target_access.json"
    previous = read_json(path) if path.exists() else []
    previous.append(
        {
            "event": len(previous) + 1,
            "selection_sha256": canonical_hash(seal),
            "purpose": "final_target_evaluation_after_source_selection",
        }
    )
    atomic_json(path, previous)
