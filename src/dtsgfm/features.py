"""Cross-view correspondence and physical graph feature modules."""

from __future__ import annotations

import torch
from torch import nn


class DelayAwareAlignment(nn.Module):
    """Align temporal view features over a physically bounded lag interval."""

    def __init__(self, hidden_dim: int, radius: int) -> None:
        super().__init__()
        self.radius = radius
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.register_buffer("reference_offsets", torch.tensor([0.0, 0.0, 1.0, 2.0]))

    def forward(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, hidden_dim, _ = features.shape
        reference = features[:, 0]
        aligned = [reference]
        offsets = [torch.zeros(batch_size, device=features.device, dtype=features.dtype)]
        shifts = torch.arange(-self.radius, self.radius + 1, device=features.device)
        scale = hidden_dim**0.5
        reference_token = reference.mean(dim=-1)
        query = self.query(reference_token)
        for view_index in range(1, num_views):
            current = features[:, view_index]
            rolled = torch.stack(
                [torch.roll(current, shifts=int(shift.item()), dims=-1) for shift in shifts],
                dim=1,
            )
            candidate_tokens = rolled.mean(dim=-1)
            keys = self.key(candidate_tokens)
            scores = torch.einsum("bh,bsh->bs", query, keys) / scale
            prior = self.reference_offsets[min(view_index, len(self.reference_offsets) - 1)]
            scores = scores - ((shifts.float() - prior) ** 2).unsqueeze(0) / 8.0
            weights = torch.softmax(scores, dim=-1)
            aligned_view = torch.einsum("bs,bshl->bhl", weights, rolled)
            availability = view_mask[:, view_index].view(batch_size, 1, 1)
            aligned.append(aligned_view * availability)
            offsets.append((weights * shifts.float().unsqueeze(0)).sum(dim=-1))
        return torch.stack(aligned, dim=1), torch.stack(offsets, dim=1)


class BayesianDirectedHypergraph(nn.Module):
    """Apply directed relation-type propagation with local Bayesian confidence."""

    edges = ((0, 1, 0), (0, 2, 1), (2, 0, 1), (0, 3, 2), (2, 3, 2))

    def __init__(
        self,
        hidden_dim: int,
        uncertainty_scale: float,
        offset_scale: float,
        use_bayesian_confidence: bool = True,
        directed: bool = True,
    ) -> None:
        super().__init__()
        self.use_bayesian_confidence = use_bayesian_confidence
        self.directed = directed
        self.uncertainty_scale = uncertainty_scale
        self.offset_scale = offset_scale
        self.relation_transforms = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(3)]
        )
        self.prior_mean = nn.Parameter(torch.full((len(self.edges),), 0.45))
        self.prior_log_variance = nn.Parameter(torch.full((len(self.edges),), -1.0))
        self.register_buffer("reference_offsets", torch.tensor([0.0, 1.0, -1.0, 2.0, 1.0]))

    def forward(
        self,
        nodes: torch.Tensor,
        quality: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, hidden_dim = nodes.shape
        propagated = torch.zeros_like(nodes)
        weight_sum = torch.zeros(batch_size, num_views, 1, device=nodes.device, dtype=nodes.dtype)
        confidence_values = []
        active_edges = list(self.edges)
        if not self.directed:
            active_edges = [
                (0, 1, 0),
                (1, 0, 0),
                (0, 2, 0),
                (2, 0, 0),
                (2, 3, 0),
                (3, 2, 0),
            ]
        for edge_index, (source, target, relation_type) in enumerate(active_edges):
            transformed = self.relation_transforms[relation_type](nodes[:, source])
            source_quality = quality[:, source]
            target_quality = quality[:, target]
            if self.directed:
                prior_index = edge_index
                prior_variance = self.prior_log_variance[prior_index].exp().clamp_min(1.0e-4)
                observed = torch.nn.functional.cosine_similarity(transformed, nodes[:, target], dim=-1)
                observation_variance = (1.25 - source_quality * target_quality).clamp_min(0.05)
                gain = prior_variance / (prior_variance + observation_variance)
                posterior_mean = self.prior_mean[prior_index] + gain * (observed - self.prior_mean[prior_index])
                posterior_variance = (1.0 - gain) * prior_variance
                offset_delta = offsets[:, target] - offsets[:, source] - self.reference_offsets[prior_index]
                confidence = torch.exp(
                    -posterior_variance / (posterior_variance + self.uncertainty_scale**2)
                ) * torch.exp(-(offset_delta**2) / (2.0 * self.offset_scale**2))
                if not self.use_bayesian_confidence:
                    confidence = torch.ones_like(confidence)
                relation_weight = confidence * torch.tanh(posterior_mean)
            else:
                confidence = source_quality * target_quality
                relation_weight = confidence
            availability = (source_quality > 0).float() * (target_quality > 0).float()
            total_weight = relation_weight * source_quality * availability
            propagated[:, target] += transformed * total_weight.unsqueeze(-1)
            weight_sum[:, target] += total_weight.abs().unsqueeze(-1)
            confidence_values.append(confidence)
        reference = torch.tanh(propagated / weight_sum.clamp_min(1.0e-6))
        confidence_matrix = torch.stack(confidence_values, dim=1)
        return reference, confidence_matrix

