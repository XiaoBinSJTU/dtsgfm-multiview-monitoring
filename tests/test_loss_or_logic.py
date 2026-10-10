"""Joint loss and backward-path checks."""

import torch

from dtsgfm import DTSGFM, load_demo_config, total_loss
from dtsgfm.preprocessing import angle_domain_align, estimate_view_quality


def test_joint_loss_and_backward() -> None:
    torch.manual_seed(13)
    config = load_demo_config()
    model = DTSGFM(config)
    model.train()
    signals = torch.randn(6, 4, config["data"]["sequence_length"])
    signals[:, 2, 20:30] = 0.0
    view_mask = torch.ones(6, 4)
    labels = torch.arange(6)
    outputs = model(signals, view_mask)
    loss, components = total_loss(outputs, labels, config)
    assert torch.isfinite(loss)
    assert set(components) == {
        "task",
        "reconstruction",
        "state_consistency",
        "physical_consistency",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_alignment_and_quality_logic() -> None:
    reference = torch.sin(torch.linspace(0.0, 2.0 * torch.pi, 64))
    signals = torch.stack([reference, torch.roll(reference, 2), reference, reference]).unsqueeze(0)
    view_mask = torch.ones(1, 4)
    aligned = angle_domain_align(signals, view_mask, max_shift=3)
    quality = estimate_view_quality(aligned, view_mask)
    assert aligned.shape == signals.shape
    assert quality.shape == view_mask.shape
    assert torch.isfinite(aligned).all()
    assert torch.all((quality > 0.0) & (quality <= 1.0))

