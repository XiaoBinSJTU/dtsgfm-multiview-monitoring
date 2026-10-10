"""Core DTSGFM model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .features import BayesianDirectedHypergraph, DelayAwareAlignment
from .preprocessing import (
    angle_domain_align,
    apply_quality_aware_mask,
    estimate_view_quality,
    same_phase_smooth_target,
)


class HierarchicalEncoder(nn.Module):
    """Compact hierarchical temporal encoder shared by all standardized views."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        middle = max(8, hidden_dim // 2)
        self.network = nn.Sequential(
            nn.Conv1d(1, middle, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(middle),
            nn.GELU(),
            nn.Conv1d(middle, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.network(signals)


@dataclass(frozen=True)
class DTSGFMOptions:
    angle_alignment: bool = True
    quality_masking: bool = True
    delay_alignment: bool = True
    directed_graph: bool = True
    bayesian_confidence: bool = True
    topology_gate: bool = True
    physical_loss: bool = True


ABLATION_OPTIONS: dict[str, DTSGFMOptions] = {
    "full": DTSGFMOptions(),
    "no_angle_alignment": DTSGFMOptions(angle_alignment=False),
    "no_quality_masking": DTSGFMOptions(quality_masking=False),
    "no_delay_alignment": DTSGFMOptions(delay_alignment=False),
    "undirected_graph": DTSGFMOptions(directed_graph=False),
    "no_bayesian_confidence": DTSGFMOptions(bayesian_confidence=False),
    "no_topology_gate": DTSGFMOptions(topology_gate=False),
    "no_physical_loss": DTSGFMOptions(physical_loss=False),
}


class DTSGFM(nn.Module):
    """Two-layer low-quality multi-view fusion model with physical propagation."""

    def __init__(self, config: dict[str, Any], variant: str = "full") -> None:
        super().__init__()
        if variant not in ABLATION_OPTIONS:
            raise ValueError(f"Unknown ablation variant: {variant}")
        self.options = ABLATION_OPTIONS[variant]
        model_config = config["model"]
        hidden_dim = int(model_config["hidden_dim"])
        self.hidden_dim = hidden_dim
        self.encoder = HierarchicalEncoder(hidden_dim, float(model_config["dropout"]))
        self.decoder = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.aligner = DelayAwareAlignment(hidden_dim, int(model_config["delay_radius"]))
        self.hypergraph = BayesianDirectedHypergraph(
            hidden_dim=hidden_dim,
            uncertainty_scale=float(model_config["relation_uncertainty_scale"]),
            offset_scale=float(model_config["relation_offset_scale"]),
            use_bayesian_confidence=self.options.bayesian_confidence,
            directed=self.options.directed_graph,
        )
        self.topology_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(0.25))
        self.gate_quality = nn.Parameter(torch.tensor(1.0))
        self.gate_deviation = nn.Parameter(torch.tensor(1.0))
        self.residual_weight = float(model_config["residual_weight"])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(model_config["dropout"])),
            nn.Linear(hidden_dim, int(config["data"]["num_classes"])),
        )
        self.mask_min = float(model_config["mask_min"])
        self.mask_max = float(model_config["mask_max"])
        self.delay_radius = int(model_config["delay_radius"])

    def _encode_views(self, signals: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, length = signals.shape
        flat = signals.reshape(batch_size * num_views, 1, length)
        encoded = self.encoder(flat)
        encoded = encoded.reshape(batch_size, num_views, self.hidden_dim, encoded.shape[-1])
        return encoded * view_mask.view(batch_size, num_views, 1, 1)

    def forward(self, signals: torch.Tensor, view_mask: torch.Tensor) -> dict[str, torch.Tensor | None]:
        aligned_input = (
            angle_domain_align(signals, view_mask, self.delay_radius)
            if self.options.angle_alignment
            else signals
        )
        quality = estimate_view_quality(aligned_input, view_mask)
        mask_indicator: torch.Tensor | None = None
        encoded = self._encode_views(aligned_input, view_mask)
        reconstruction_features = encoded
        if self.training and self.options.quality_masking:
            masked_input, mask_indicator = apply_quality_aware_mask(
                aligned_input, quality, self.mask_min, self.mask_max
            )
            reconstruction_features = self._encode_views(masked_input, view_mask)
        reconstruction = self.decoder(
            reconstruction_features.reshape(
                -1, self.hidden_dim, reconstruction_features.shape[-1]
            )
        )
        reconstruction = F.interpolate(
            reconstruction,
            size=signals.shape[-1],
            mode="nearest",
        ).reshape_as(signals)
        if self.options.delay_alignment:
            encoded, offsets = self.aligner(encoded, view_mask)
        else:
            offsets = torch.zeros(signals.shape[0], signals.shape[1], device=signals.device)
        nodes = encoded.mean(dim=-1)
        physical_reference, relation_confidence = self.hypergraph(nodes, quality, offsets)
        deviation = ((nodes - physical_reference) ** 2).mean(dim=-1)
        if self.options.topology_gate:
            gate = torch.sigmoid(
                self.gate_bias + self.gate_quality * quality - self.gate_deviation.abs() * deviation
            )
            residual = torch.tanh(self.topology_residual(nodes - physical_reference))
            fused_nodes = (
                gate.unsqueeze(-1) * nodes
                + (1.0 - gate).unsqueeze(-1) * physical_reference
                + self.residual_weight * gate.unsqueeze(-1) * residual
            )
        else:
            gate = torch.full_like(quality, 0.5)
            fused_nodes = nodes + 0.35 * physical_reference
        node_weights = quality / quality.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        fused = (fused_nodes * node_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "quality": quality,
            "offsets": offsets,
            "nodes": nodes,
            "physical_reference": physical_reference,
            "fused_nodes": fused_nodes,
            "relation_confidence": relation_confidence,
            "topology_gate": gate,
            "reconstruction": reconstruction,
            "reconstruction_target": same_phase_smooth_target(aligned_input),
            "mask_indicator": mask_indicator,
        }

