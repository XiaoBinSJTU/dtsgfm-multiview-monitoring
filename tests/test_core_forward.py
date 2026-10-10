"""Core model forward and constraint checks."""

import torch

from dtsgfm import DTSGFM, load_demo_config


def test_core_forward_is_finite() -> None:
    torch.manual_seed(11)
    config = load_demo_config()
    model = DTSGFM(config)
    model.eval()
    signals = torch.randn(4, config["data"]["num_views"], config["data"]["sequence_length"])
    view_mask = torch.ones(4, config["data"]["num_views"])
    view_mask[0, 3] = 0.0
    signals[0, 3] = 0.0
    with torch.no_grad():
        outputs = model(signals, view_mask)
    logits = outputs["logits"]
    quality = outputs["quality"]
    gate = outputs["topology_gate"]
    assert logits is not None and quality is not None and gate is not None
    assert logits.shape == (4, config["data"]["num_classes"])
    assert torch.isfinite(logits).all()
    assert torch.all((quality >= 0.0) & (quality <= 1.0))
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
    assert quality[0, 3].item() == 0.0

