"""Audit source-bound outcomes, preserved legacy contracts, pinned references, and code style."""

import argparse
import ast
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification"), str(ROOT / "office_home")]


def main():
    """Require successful outcomes instead of treating generated report existence as evidence."""
    from clvision.config import MAIN_METHODS
    from clvision.identity import source_hash
    from ohvision.fingerprint import source_hash as office_hash

    from cue_routing.io import atomic_json, file_hash, read_json
    from cue_routing.io import source_hash as legacy_hash

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lint", action="store_true")
    arguments = parser.parse_args()
    fingerprint = source_hash(ROOT)
    extension = ROOT / "classification"
    evidence = {}
    names = (
        "unit_tests",
        "cpu_smoke",
        "cifar100_cpu_smoke",
        "tiny_imagenet_cpu_smoke",
        "dominoes_cpu_smoke",
        "dataset_preparation",
    )
    for name in names:
        path = extension / "verification" / f"{name}.json"
        value = read_json(path) if path.exists() else {}
        valid = value.get("source_sha256") == fingerprint
        if name == "unit_tests":
            valid = (
                valid
                and value.get("passed") is True
                and value.get("tests_run", 0) >= 121
                and all(value.get(key) == 0 for key in ("failures", "errors", "skipped"))
                and value.get("environment", {}).get("environment_manager") == "conda"
            )
        elif name == "cpu_smoke":
            valid = (
                valid
                and value.get("passed") is True
                and value.get("benchmark_evidence") is False
                and value.get("completed_study_reused") is True
                and value.get("reordered_predictions_rejected") is True
            )
        elif name == "dataset_preparation":
            datasets = value.get("datasets", {})
            valid = (
                valid
                and value.get("target_predictions_computed") is False
                and set(datasets) == {"cifar100", "tiny_imagenet", "dominoes", "cifar100_c"}
                and all(item.get("verified") is True for item in datasets.values())
            )
        else:
            methods = value.get("methods", {})
            valid = (
                valid
                and value.get("benchmark_evidence") is False
                and value.get("target_loader_accessed") is False
                and value.get("matched_initial_weights") is True
                and value.get("matched_augmented_batches") is True
                and set(methods) == set(MAIN_METHODS)
                and all(
                    item.get("backbone_updated") is True and item.get("optimizer_steps", 0) >= 1
                    for item in methods.values()
                )
            )
        evidence[name] = {"exists": path.exists(), "outcome_and_source_valid": bool(valid)}
    comments = []
    for path in extension.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        content = path.read_text()
        ast.parse(content, filename=str(path))
        comments.extend(
            {"file": str(path.relative_to(ROOT)), "line": token.start[0]}
            for token in tokenize.generate_tokens(io.StringIO(content).readline)
            if token.type == tokenize.COMMENT
        )
    provenance = read_json(extension / "references/zipfls_provenance.json")
    references_valid = all(
        file_hash(ROOT / item["file"]) == item["sha256"] for item in provenance["files"]
    )
    lint = []
    if arguments.lint:
        for command in (("check", "classification"), ("format", "--check", "classification")):
            result = subprocess.run(
                [sys.executable, "-m", "ruff", *command], cwd=ROOT, capture_output=True, text=True
            )
            lint.append(
                {
                    "command": list(command),
                    "exit_code": result.returncode,
                    "output": result.stdout + result.stderr,
                }
            )
    result = {
        "source_sha256": fingerprint,
        "legacy_core_unchanged": legacy_hash(ROOT)
        == "583fc84119e4d5ff16b323e91d446f394bd824eb1f400a94ee8bfdadff9ad4e2",
        "officehome_unchanged": office_hash(ROOT)
        == "273d238783d9b723e0e74604bad856b3209e485d4af8b4115aab5c82698bb95b",
        "reference_checksums_valid": references_valid,
        "line_comments": comments,
        "evidence": evidence,
        "lint": lint,
        "cuda_fit_measured_on_this_host": all(
            (extension / "verification" / f"{name}_cuda.json").exists()
            and read_json(extension / "verification" / f"{name}_cuda.json").get("passes_budget")
            is True
            and read_json(extension / "verification" / f"{name}_cuda.json").get("source_sha256")
            == fingerprint
            for name in ("cifar100", "tiny_imagenet", "dominoes")
        ),
        "benchmark_efficacy_assessed_by_this_audit": False,
    }
    result["passed"] = (
        result["legacy_core_unchanged"]
        and result["officehome_unchanged"]
        and references_valid
        and not comments
        and all(all(item.values()) for item in evidence.values())
        and all(item["exit_code"] == 0 for item in lint)
    )
    atomic_json(extension / "verification/audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
