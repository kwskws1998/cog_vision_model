"""Dataset-level classification metrics computed from unmasked predictions."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_metrics(logits, targets, bins=15):
    """Compute top-1, macro recall, NLL, Brier score, ECE, and confusion counts."""
    if targets.dtype != torch.long:
        raise ValueError("Metrics require int64 class labels")
    logits, targets = logits.detach().double().cpu(), targets.detach().cpu()
    if logits.ndim != 2 or targets.shape != (len(logits),) or len(logits) == 0 or bins <= 0:
        raise ValueError("Metrics require nonempty NC logits and N labels")
    if logits.shape[1] < 2:
        raise ValueError("Metrics require at least two classes")
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError(
            f"Labels outside [0, {logits.shape[1] - 1}]: "
            f"min={int(targets.min())}, max={int(targets.max())}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError(
            f"Nonfinite logits: nan={int(logits.isnan().sum())}, "
            f"inf={int(logits.isinf().sum())}, shape={tuple(logits.shape)}"
        )
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = (confidence >= lower) & (
            (confidence < upper) if index + 1 < bins else (confidence <= upper)
        )
        if members.any():
            ece += float(
                members.double().mean()
                * (confidence[members].mean() - correct[members].double().mean()).abs()
            )
    classes = logits.shape[1]
    confusion = torch.bincount(targets * classes + prediction, minlength=classes * classes).reshape(
        classes, classes
    )
    present = confusion.sum(dim=1) > 0
    macro_recall = (confusion.diag()[present].double() / confusion.sum(dim=1)[present]).mean()
    return {
        "n": len(targets),
        "accuracy": float(correct.double().mean()),
        "balanced_accuracy": float(macro_recall),
        "nll": float(F.cross_entropy(logits, targets)),
        "brier": float((probabilities - F.one_hot(targets, classes)).square().sum(dim=1).mean()),
        "ece": ece,
        "ece_bins": bins,
        "confusion": confusion.tolist(),
    }


def aggregate_domains(results):
    """Give each source validation domain equal model-selection weight."""
    if not results:
        raise ValueError("No domain results to aggregate")
    return {
        "domains": results,
        "macro_accuracy": sum(item["accuracy"] for item in results.values()) / len(results),
        "macro_nll": sum(item["nll"] for item in results.values()) / len(results),
        "n": sum(item["n"] for item in results.values()),
    }
