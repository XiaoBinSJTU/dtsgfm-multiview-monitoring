"""Task and auxiliary losses for model training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def reconstruction_loss(outputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    """Compute masked robust reconstruction loss when a mask is active."""
    reconstruction = outputs.get("reconstruction")
    target = outputs.get("reconstruction_target")
    indicator = outputs.get("mask_indicator")
    if reconstruction is None or target is None or indicator is None or not indicator.any():
        logits = outputs["logits"]
        assert logits is not None
        return logits.sum() * 0.0
    errors = F.smooth_l1_loss(reconstruction, target, reduction="none")
    return errors[indicator].mean()


def state_consistency_loss(outputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    """Penalize quality-weighted disagreement among aligned view states."""
    nodes = outputs.get("nodes")
    quality = outputs.get("quality")
    if nodes is None or quality is None:
        logits = outputs["logits"]
        assert logits is not None
        return logits.sum() * 0.0
    weighted_mean = (nodes * quality.unsqueeze(-1)).sum(dim=1) / quality.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    disagreement = ((nodes - weighted_mean.unsqueeze(1)) ** 2).mean(dim=-1)
    return (disagreement * quality).sum() / quality.sum().clamp_min(1.0)


def physical_consistency_loss(outputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
    """Constrain low-confidence fused states toward their physical references."""
    fused = outputs.get("fused_nodes")
    reference = outputs.get("physical_reference")
    quality = outputs.get("quality")
    if fused is None or reference is None or quality is None:
        logits = outputs["logits"]
        assert logits is not None
        return logits.sum() * 0.0
    error = ((fused - reference) ** 2).mean(dim=-1)
    weights = 1.0 - quality
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def total_loss(
    outputs: dict[str, torch.Tensor | None],
    labels: torch.Tensor,
    config: dict[str, Any],
    include_physical: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine classification, reconstruction, state, and physical losses."""
    logits = outputs["logits"]
    assert logits is not None
    task = F.cross_entropy(logits, labels)
    reconstruction = reconstruction_loss(outputs)
    state = state_consistency_loss(outputs)
    physical = physical_consistency_loss(outputs) if include_physical else task * 0.0
    weights = config["training"]["loss_weights"]
    total = (
        task
        + float(weights["reconstruction"]) * reconstruction
        + float(weights["state_consistency"]) * state
        + float(weights["physical_consistency"]) * physical
    )
    values = {
        "task": float(task.detach().cpu()),
        "reconstruction": float(reconstruction.detach().cpu()),
        "state_consistency": float(state.detach().cpu()),
        "physical_consistency": float(physical.detach().cpu()),
    }
    return total, values

