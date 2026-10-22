"""Detached cue-wise categorical error routing with an exact autograd surrogate."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def validate_cues(cue_logits, weights):
    """Enforce normalized full-precision cue inputs for both public routing functions."""
    if (
        cue_logits.ndim != 3
        or min(cue_logits.shape) < 1
        or cue_logits.shape[-1] < 2
        or weights.ndim != 1
        or cue_logits.shape[1] != weights.numel()
    ):
        raise ValueError("Expected nonempty BKC cue logits and K weights")
    if (
        cue_logits.dtype not in (torch.float32, torch.float64)
        or weights.dtype != cue_logits.dtype
        or weights.device != cue_logits.device
    ):
        raise ValueError("Cue logits and weights must share FP32/FP64 dtype and device")
    if not torch.isfinite(cue_logits).all() or not torch.isfinite(weights).all():
        raise ValueError("Nonfinite routing input")
    if (weights <= 0).any() or not torch.isclose(
        weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0
    ):
        raise ValueError("Cue weights must be positive and sum to one")


def route_gates(
    cue_logits, weights, tau=0.15, temperature=0.05, joint_floor=0.5, mode="cue", fixed_joint=0.75
):
    """Measure probability disagreement and detach the entire strategy decision."""
    validate_cues(cue_logits, weights)
    if (
        not all(math.isfinite(value) for value in (tau, temperature, joint_floor, fixed_joint))
        or temperature <= 0
        or not 0 <= joint_floor <= 1
        or not 0 <= tau <= 1
    ):
        raise ValueError("Invalid routing hyperparameters")
    if not 0 <= fixed_joint <= 1 or mode not in ("cue", "image", "fixed", "shuffled"):
        raise ValueError("Invalid routing mode or fixed joint fraction")
    with torch.no_grad():
        joint_logits = (cue_logits * weights[None, :, None]).sum(dim=1)
        local_probabilities = cue_logits.softmax(dim=-1)
        joint_probabilities = joint_logits.softmax(dim=-1)
        disagreement = (local_probabilities - joint_probabilities[:, None]).norm(
            dim=-1
        ) / math.sqrt(2.0)
        if mode == "image":
            disagreement_for_gate = (
                (disagreement.square() * weights[None])
                .sum(dim=1, keepdim=True)
                .sqrt()
                .expand_as(disagreement)
            )
        else:
            disagreement_for_gate = disagreement
        gates = joint_floor + (1.0 - joint_floor) * torch.sigmoid(
            (disagreement_for_gate - tau) / temperature
        )
        if mode == "fixed":
            gates = torch.full_like(gates, fixed_joint)
        elif mode == "shuffled":
            gates = gates.roll(shifts=1, dims=1)
    return gates, disagreement, joint_logits


def routed_loss(
    cue_logits,
    targets,
    weights,
    tau=0.15,
    temperature=0.05,
    joint_floor=0.5,
    mode="cue",
    fixed_joint=0.75,
    gates_override=None,
):
    """Route each cue's joint gradient locally without averaging away its gate."""
    if (
        cue_logits.ndim != 3
        or targets.shape != (cue_logits.shape[0],)
        or targets.dtype != torch.long
        or targets.device != cue_logits.device
    ):
        raise ValueError("Expected one int64 class label per image")
    if (targets < 0).any() or (targets >= cue_logits.shape[-1]).any():
        raise ValueError("Target is outside the class range")
    gates, disagreement, joint_logits = route_gates(
        cue_logits, weights, tau, temperature, joint_floor, mode, fixed_joint
    )
    if gates_override is not None:
        if (
            gates_override.device != cue_logits.device
            or gates_override.shape != gates.shape
            or not torch.isfinite(gates_override).all()
            or (gates_override < 0).any()
            or (gates_override > 1).any()
        ):
            raise ValueError("Invalid explicitly supplied gates")
        gates = gates_override.detach()
    surrogate = joint_logits.detach()[:, None, :] + (cue_logits - cue_logits.detach())
    expanded_targets = targets[:, None].expand(cue_logits.shape[:2]).reshape(-1)
    joint_terms = F.cross_entropy(
        surrogate.reshape(-1, cue_logits.shape[-1]), expanded_targets, reduction="none"
    ).reshape_as(gates)
    marginal_terms = F.cross_entropy(
        cue_logits.reshape(-1, cue_logits.shape[-1]), expanded_targets, reduction="none"
    ).reshape_as(gates)
    loss = (
        ((gates * joint_terms + (1.0 - gates) * marginal_terms) * weights[None]).sum(dim=1).mean()
    )
    diagnostics = {
        "joint_ce": float(F.cross_entropy(joint_logits.detach(), targets)),
        "marginal_ce": float((marginal_terms.detach() * weights[None]).sum(dim=1).mean()),
        "joint_fraction": float((gates * weights[None]).sum(dim=1).mean()),
        "disagreement": float((disagreement * weights[None]).sum(dim=1).mean()),
        "gate_min": float(gates.min()),
        "gate_max": float(gates.max()),
    }
    return loss, joint_logits.detach(), diagnostics
