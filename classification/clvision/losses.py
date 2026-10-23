"""Matched regularization controls and a device-independent port of pinned Zipf loss."""

import torch
from torch.nn import functional as F

from cue_routing.engine import autocast_context
from cue_routing.routing import route_gates


def zipf_distribution(votes, targets, classes, power=1.0):
    """Match SciPy minimum-tie descending ranks, excluding and renormalizing the target."""
    if votes.ndim != 2 or votes.dtype != torch.long or targets.shape != votes.shape[:1]:
        raise ValueError("Zipf votes must be long BN with one target per image")
    if classes < 2 or votes.shape[1] == 0 or (votes < 0).any() or (votes >= classes).any():
        raise ValueError("Invalid votes or class count")
    if (
        targets.dtype != torch.long
        or (targets < 0).any()
        or (targets >= classes).any()
        or power <= 0
    ):
        raise ValueError("Invalid targets or Zipf exponent")
    with torch.no_grad():
        counts = torch.zeros(len(votes), classes, device=votes.device, dtype=torch.long)
        counts.scatter_add_(1, votes, torch.ones_like(votes))
        ranks = 1 + (counts[:, None, :] > counts[:, :, None]).sum(-1)
        mask = ~F.one_hot(targets, classes).bool()
        values = ranks.double().pow(-power)[mask].reshape(len(votes), classes - 1)
        return values / values.sum(-1, keepdim=True)


def zipf_kl(logits, votes, targets, power=1.0):
    """Match unmasked official KL(q_zipf || normalized non-target predictions)."""
    distribution = zipf_distribution(votes, targets, logits.shape[1], power).to(logits.dtype)
    mask = ~F.one_hot(targets, logits.shape[1]).bool()
    nontarget_logits = logits[mask].reshape(len(logits), logits.shape[1] - 1)
    return F.kl_div(nontarget_logits.log_softmax(-1), distribution, reduction="none").sum(-1).mean()


def consistency_loss(local_logits, areas, targets, config):
    """Realize the exact CUE gradient and coefficient-matched image and permutation controls."""
    joint = (local_logits * areas[None, :, None]).sum(1)
    gates, disagreement, _ = route_gates(
        local_logits, areas, config.gate_tau, config.gate_temperature, config.joint_floor
    )
    if config.method == "fixed_global_local":
        gates = torch.full_like(gates, 1 - config.fixed_lambda)
    elif config.method == "pure_marginal":
        gates = torch.zeros_like(gates)
    elif config.method == "image_mean_gate":
        gates = (gates * areas[None]).sum(1, keepdim=True).expand_as(gates)
    elif config.method == "permuted_gate":
        gates = gates[:, [2, 3, 0, 1]]
    teacher = joint.softmax(-1).detach()
    local_kl = F.kl_div(
        local_logits.log_softmax(-1), teacher[:, None].expand_as(local_logits), reduction="none"
    ).sum(-1)
    regularizer = (areas[None] * (1 - gates) * local_kl).sum(1).mean()
    global_ce = F.cross_entropy(joint, targets)
    mean_gate = (gates * areas[None]).sum(1)
    gate_variance = (areas[None] * (gates - mean_gate[:, None]).square()).sum(1)
    diagnostics = {
        "global_ce": global_ce.detach(),
        "local_consistency": regularizer.detach(),
        "joint_fraction": mean_gate.mean().detach(),
        "gate_within_image_std": gate_variance.sqrt().mean().detach(),
        "joint_saturated_fraction": (areas[None] * (gates >= 0.99).to(gates.dtype))
        .sum(1)
        .mean()
        .detach(),
        "gate_min": gates.min().detach(),
        "gate_max": gates.max().detach(),
        "disagreement": (disagreement * areas[None]).sum(1).mean().detach(),
    }
    return global_ce + regularizer, joint, diagnostics


def training_objective(model, images, targets, config):
    """Perform one encoder forward with an explicitly selected training intervention."""
    with autocast_context(config):
        penultimate, final = model.feature_maps(images)
    if config.method == "zipfls_dense2":
        logits, auxiliary, votes = model.zipf_predictions(penultimate, final)
        auxiliary_ce = F.cross_entropy(auxiliary, targets)
        distillation = zipf_kl(logits, votes, targets, config.zipf_power)
        loss = (
            F.cross_entropy(logits, targets)
            + config.zipf_aux_weight * auxiliary_ce
            + config.zipf_lambda * distillation
        )
        diagnostics = {
            "zipf_kl": distillation.detach(),
            "auxiliary_ce": auxiliary_ce.detach(),
        }
    elif config.method in ("erm", "label_smoothing"):
        logits = model.readout(final)
        if config.method == "label_smoothing":
            log_probabilities = logits.log_softmax(-1)
            hard = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
            non_target = (-log_probabilities.sum(1) - hard) / (logits.shape[1] - 1)
            loss = (
                (1 - config.label_smoothing) * hard + config.label_smoothing * non_target
            ).mean()
        else:
            loss = F.cross_entropy(logits, targets)
        diagnostics = {}
    else:
        local_logits, areas = model.readout.cue_logits(final)
        loss, logits, diagnostics = consistency_loss(local_logits, areas, targets, config)
    diagnostics["train_accuracy"] = logits.detach().argmax(1).eq(targets).float().mean()
    return loss, diagnostics
