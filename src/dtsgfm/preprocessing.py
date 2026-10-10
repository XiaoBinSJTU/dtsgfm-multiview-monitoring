"""Signal alignment, quality estimation, and continuous masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def estimate_view_quality(signals: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    """Estimate confidence from missingness, roughness, and amplitude instability."""
    zero_fraction = (signals.abs() < 1.0e-7).float().mean(dim=-1)
    roughness = (signals[..., 1:] - signals[..., :-1]).abs().mean(dim=-1)
    amplitude = signals.abs().mean(dim=-1)
    amplitude_penalty = (amplitude - 0.65).abs()
    raw_penalty = 2.2 * zero_fraction + 0.28 * roughness + 0.18 * amplitude_penalty
    quality = torch.exp(-raw_penalty).clamp(0.02, 1.0)
    return quality * view_mask.float()


def angle_domain_align(
    signals: torch.Tensor,
    view_mask: torch.Tensor,
    max_shift: int = 3,
) -> torch.Tensor:
    """Align available views to the mechanical reference by bounded correlation."""
    if max_shift <= 0:
        return signals
    reference = signals[:, 0]
    aligned = signals.clone()
    for view_index in range(1, signals.shape[1]):
        candidate_scores = []
        candidates = []
        current = signals[:, view_index]
        for shift in range(-max_shift, max_shift + 1):
            shifted = torch.roll(current, shifts=shift, dims=-1)
            score = F.cosine_similarity(reference, shifted, dim=-1)
            candidate_scores.append(score)
            candidates.append(shifted)
        score_stack = torch.stack(candidate_scores, dim=-1)
        signal_stack = torch.stack(candidates, dim=1)
        best = score_stack.argmax(dim=-1)
        batch_index = torch.arange(signals.shape[0], device=signals.device)
        selected = signal_stack[batch_index, best]
        availability = view_mask[:, view_index].unsqueeze(-1)
        aligned[:, view_index] = selected * availability
    return aligned


def apply_quality_aware_mask(
    signals: torch.Tensor,
    quality: torch.Tensor,
    minimum: float,
    maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask one continuous segment per available view using quality-dependent ratios."""
    batch_size, num_views, length = signals.shape
    ratios = minimum + (maximum - minimum) * (1.0 - quality)
    masked = signals.clone()
    indicator = torch.zeros_like(signals, dtype=torch.bool)
    for batch_index in range(batch_size):
        for view_index in range(num_views):
            if quality[batch_index, view_index] <= 0:
                continue
            segment_length = int(round(float(ratios[batch_index, view_index].detach()) * length))
            segment_length = max(1, min(segment_length, length - 1))
            upper = length - segment_length + 1
            start = int(torch.randint(0, upper, (1,), device=signals.device).item())
            stop = start + segment_length
            masked[batch_index, view_index, start:stop] = 0.0
            indicator[batch_index, view_index, start:stop] = True
    return masked, indicator


def same_phase_smooth_target(signals: torch.Tensor) -> torch.Tensor:
    """Build a robust local target that suppresses isolated disturbances."""
    flat = signals.reshape(-1, 1, signals.shape[-1])
    padded = F.pad(flat, (2, 2), mode="reflect")
    smooth = F.avg_pool1d(padded, kernel_size=5, stride=1)
    return smooth.reshape_as(signals)

