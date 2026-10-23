"""Audit current source-bound Office-Home evidence and preserve historical compatibility status."""

import argparse
import ast
import io
import json
import math
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "office_home"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXTENSION))

from ohvision.config import CLASSES, DOMAINS, EXPECTED_COUNTS, METHODS, ExperimentConfig
from ohvision.fingerprint import source_hash

from cue_routing.io import atomic_json, read_json
from cue_routing.io import source_hash as core_source_hash


def valid_report(name, value):
    """Require successful semantic outcomes, not just a matching source fingerprint."""
    if name == "unit_tests":
        return (
            value.get("passed") is True
            and value.get("tests_run", 0) >= 76
            and all(value.get(key) == 0 for key in ("errors", "failures", "skipped"))
            and value.get("environment", {}).get("environment_manager") == "conda"
        )
    if name == "static_checks":
        return value.get("passed") is True
    if name == "dataset_preparation":
        return (
            value.get("images") == 15588
            and value.get("classes") == 65
            and value.get("variant") == "original_resolution"
            and value.get("domain_counts") == EXPECTED_COUNTS
            and value.get("full_rgb_decoding_passed") is True
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
            and item.get("target_access_events") == 1
            for item in methods.values()
        )
    valid = all(
        item.get("backbone_updated") is True
        and math.isfinite(item.get("diagnostics", {}).get("loss", float("nan")))
        for item in methods.values()
    )
    if name == "pretrained_cpu_smoke":
        return valid and all(item.get("parameter_count") == 11209857 for item in methods.values())
    if name == "real_data_cpu_smoke":
        return (
            valid
            and value.get("verified_image_count") == 15588
            and value.get("matched_training_images_and_augmentations") is True
            and value.get("target_loader_accessed") is False
            and all(
                item.get("optimizer_steps") == value.get("optimizer_steps_per_method")
                and len(item.get("source_validation_domains_checked", [])) == 3
                for item in methods.values()
            )
        )
    return False


def main():
    """Generate an audit, optionally recording fresh Ruff checks inside the same Conda env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lint", action="store_true")
    arguments = parser.parse_args()
    fingerprint = source_hash(ROOT)
    compatibility = read_json(EXTENSION / "references/pacs_compatibility.json")
    current_core = core_source_hash(ROOT)
    paths = [path for path in EXTENSION.rglob("*.py") if "__pycache__" not in path.parts]
    comments = []
    for path in paths:
        content = path.read_text()
        ast.parse(content, filename=str(path))
        comments.extend(
            {"file": str(path.relative_to(ROOT)), "line": token.start[0]}
            for token in tokenize.generate_tokens(io.StringIO(content).readline)
            if token.type == tokenize.COMMENT
        )
    if arguments.lint:
        checks = []
        for command in (["check", "office_home"], ["format", "--check", "office_home"]):
            result = subprocess.run(
                [sys.executable, "-m", "ruff", *command], cwd=ROOT, capture_output=True, text=True
            )
            checks.append(
                {
                    "command": ["ruff", *command],
                    "exit_code": result.returncode,
                    "output": result.stdout + result.stderr,
                }
            )
        atomic_json(
            EXTENSION / "verification/static_checks.json",
            {
                "source_sha256": fingerprint,
                "passed": all(item["exit_code"] == 0 for item in checks),
                "checks": checks,
            },
        )
    evidence = {}
    for name in (
        "unit_tests",
        "cpu_smoke",
        "pretrained_cpu_smoke",
        "real_data_cpu_smoke",
        "dataset_preparation",
        "static_checks",
    ):
        path = EXTENSION / "verification" / f"{name}.json"
        value = read_json(path) if path.exists() else {}
        evidence[name] = {
            "exists": path.exists(),
            "matches_current_source": value.get("source_sha256") == fingerprint,
            "outcome_valid": valid_report(name, value),
        }
    recipe = ExperimentConfig.load(EXTENSION / "configs/officehome_resnet18.json")
    result = {
        "source_sha256": fingerprint,
        "shared_core_source_sha256": current_core,
        "pacs_fingerprint_unchanged": current_core == compatibility["shared_core_source_sha256"],
        "recipe_matches_validated_defaults": recipe.to_dict()
        == ExperimentConfig().validate().to_dict(),
        "classes": len(CLASSES),
        "domains": list(DOMAINS),
        "python_files": len(paths),
        "line_comments": comments,
        "verification": evidence,
        "cuda_fit_verified": False,
        "cuda_pretrained_source_steps_verified": False,
    }
    memory_path = EXTENSION / "verification/cuda_memory.json"
    if memory_path.exists():
        memory = read_json(memory_path)
        result["cuda_fit_verified"] = (
            memory.get("measured") is True
            and memory.get("passes_budget") is True
            and memory.get("source_sha256") == fingerprint
            and memory.get("config") == recipe.to_dict()
        )
    cuda_path = EXTENSION / "verification/real_data_cuda_smoke.json"
    if cuda_path.exists():
        cuda = read_json(cuda_path)
        result["cuda_pretrained_source_steps_verified"] = (
            cuda.get("source_sha256") == fingerprint
            and cuda.get("cuda_memory_measured") is True
            and cuda.get("optimizer_steps_per_method", 0) >= 10
            and cuda.get("target_loader_accessed") is False
            and set(cuda.get("methods", {})) == set(METHODS)
            and all(
                item.get("backbone_updated") is True and item.get("optimizer_steps", 0) >= 10
                for item in cuda["methods"].values()
            )
        )
    result["passed"] = (
        result["recipe_matches_validated_defaults"]
        and not comments
        and all(all(item.values()) for item in evidence.values())
    )
    atomic_json(EXTENSION / "verification/audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
