"""RSC adapted from pinned DomainBed code; see references/DOMAINBED_LICENSE.txt."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def rsc_logits(features, targets, classifier, feature_drop=1.0 / 3.0, batch_drop=1.0 / 3.0):
    """Match DomainBed's signed-gradient and batch-percentile RSC masks."""
    if (
        features.ndim != 2
        or min(features.shape) < 1
        or targets.shape != (features.shape[0],)
        or targets.dtype != torch.long
    ):
        raise ValueError("RSC requires a BD feature matrix and B targets")
    if not torch.isfinite(features).all() or features.device != targets.device:
        raise ValueError("RSC requires finite features and colocated labels")
    if features.dtype != classifier.weight.dtype:
        raise ValueError("RSC features must use the classifier's full-precision dtype")
    if (targets < 0).any() or (targets >= classifier.out_features).any():
        raise ValueError("RSC target outside class range")
    if not features.requires_grad:
        features = features.detach().requires_grad_(True)
    if not 0 < feature_drop < 1 or not 0 < batch_drop < 1:
        raise ValueError("RSC drop fractions must be strictly between zero and one")
    with torch.autocast(device_type=features.device.type, enabled=False):
        logits = classifier(features)
        selected = logits.gather(1, targets[:, None]).sum()
        gradients = torch.autograd.grad(selected, features, create_graph=False)[0]
        with torch.no_grad():
            thresholds = torch.quantile(
                gradients.double(), 1.0 - feature_drop, dim=1, keepdim=True
            ).to(gradients.dtype)
            feature_mask = gradients < thresholds
            muted_logits = classifier(features * feature_mask.to(features.dtype))
            changes = F.softmax(logits.detach(), dim=1).gather(1, targets[:, None]).squeeze(
                1
            ) - F.softmax(muted_logits, dim=1).gather(1, targets[:, None]).squeeze(1)
            batch_threshold = torch.quantile(changes.double(), 1.0 - batch_drop).to(changes.dtype)
            batch_mask = changes < batch_threshold
            mask = torch.logical_or(feature_mask, batch_mask[:, None])
        challenged = classifier(features * mask.to(features.dtype))
    diagnostics = {
        "rsc_kept_fraction": float(mask.float().mean()),
        "rsc_challenged_samples": float((~batch_mask).float().mean()),
        "rsc_probability_drop": float(changes.mean()),
    }
    return challenged, logits.detach(), mask, diagnostics
