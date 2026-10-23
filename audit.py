"""Audit syntax, experiment contracts, pinned splits, and current verification evidence."""

import ast
import io
import math
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cue_routing.config import METHODS, ExperimentConfig
from cue_routing.data import read_official_splits
from cue_routing.io import atomic_json, file_hash, read_json, source_hash


def valid_evidence(name, value):
    """Require successful outcomes as well as a matching fingerprint in saved reports."""
    if name == "unit_tests":
        return (
            value.get("passed") is True
            and value.get("tests_run", 0) > 0
            and all(value.get(key) == 0 for key in ("failures", "errors", "skipped"))
            and value.get("environment", {}).get("environment_manager") == "conda"
        )
    if name == "static_checks":
        return value.get("passed") is True
    if name == "dataset_preparation":
        return (
            value.get("images") == 9991
            and value.get("full_rgb_decoding_passed") is True
            and value.get("official_split_blob_count") == 12
            and value.get("target_predictions_computed") is False
        )
    methods = value.get("methods", {})
    if (
        set(methods) != set(METHODS)
        or value.get("benchmark_evidence") is not False
        or value.get("matched_initial_weights") is not True
        or value.get("environment", {}).get("environment_manager") != "conda"
    ):
        return False
    if name == "cpu_smoke":
        return all(
            item.get("training", {}).get("status") == "complete"
            and item.get("training", {}).get("benchmark_evidence") is False
            and item.get("target_access_events") == 1
            for item in methods.values()
        )
    passed = all(
        item.get("backbone_updated") is True
        and math.isfinite(item.get("diagnostics", {}).get("loss", float("nan")))
        for item in methods.values()
    )
    if name == "real_data_cpu_smoke":
        passed = (
            passed
            and value.get("verified_image_count") == 9991
            and value.get("matched_training_images_and_augmentations") is True
            and value.get("target_loader_accessed") is False
        )
    return passed


def main():
    """Fail on stale evidence and report actual resource limits without extrapolation."""
    paths = (
        list(ROOT.glob("*.py"))
        + list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
    )
    functions, comments = 0, []
    for path in paths:
        content = path.read_text()
        tree = ast.parse(content, filename=str(path))
        functions += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
        )
        comments.extend(
            {"file": str(path.relative_to(ROOT)), "line": token.start[0]}
            for token in tokenize.generate_tokens(io.StringIO(content).readline)
            if token.type == tokenize.COMMENT
        )
    recipe = ExperimentConfig.load(ROOT / "configs" / "pacs_resnet18.json")
    if recipe.to_dict() != ExperimentConfig().validate().to_dict():
        raise AssertionError("Primary JSON and validated defaults differ")
    splits, hashes = read_official_splits(ROOT / "references" / "pacs_splits")
    fingerprint = source_hash(ROOT)
    evidence = {}
    for name in (
        "unit_tests",
        "cpu_smoke",
        "pretrained_cpu_smoke",
        "real_data_cpu_smoke",
        "dataset_preparation",
        "static_checks",
    ):
        path = ROOT / "verification" / f"{name}.json"
        value = read_json(path) if path.exists() else {}
        evidence[name] = {
            "exists": path.exists(),
            "matches_current_source": value.get("source_sha256") == fingerprint,
            "outcome_valid": valid_evidence(name, value),
            "sha256": file_hash(path) if path.exists() else None,
        }
    result = {
        "syntax_valid": True,
        "python_files": len(paths),
        "functions": functions,
        "line_comments": comments,
        "source_sha256": fingerprint,
        "official_split_files": len(hashes),
        "official_image_count": sum(len(domain["test"]) for domain in splits.values()),
        "verification": evidence,
        "cuda_fit_verified": False,
    }
    memory_path = ROOT / "verification" / "cuda_memory.json"
    if memory_path.exists():
        memory = read_json(memory_path)
        result["cuda_fit_verified"] = bool(
            memory.get("measured")
            and memory.get("passes_budget")
            and memory.get("source_sha256") == fingerprint
            and memory.get("config") == recipe.to_dict()
        )
    setup_path = ROOT / "verification" / "conda_cpu_setup.json"
    setup = read_json(setup_path) if setup_path.exists() else {}
    result["conda_setup_verified"] = (
        setup.get("environment_manager") == "conda"
        and setup.get("requested_device") == "cpu"
        and setup.get("environment_yml_sha256") == file_hash(ROOT / "environment.yml")
        and setup.get("bootstrap_sha256") == file_hash(ROOT / "bootstrap.py")
        and setup.get("project_metadata_sha256") == file_hash(ROOT / "pyproject.toml")
        and setup.get("requirements_sha256")
        == {path.name: file_hash(path) for path in sorted(ROOT.glob("requirements*.txt"))}
    )
    result["passed"] = (
        not comments
        and result["conda_setup_verified"]
        and all(
            item["exists"] and item["matches_current_source"] and item["outcome_valid"]
            for item in evidence.values()
        )
    )
    atomic_json(ROOT / "verification" / "audit.json", result)
    import json

    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
