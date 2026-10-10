"""Run a forward, loss, and backward smoke test on synthetic inputs only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dtsgfm import DTSGFM, load_demo_config, total_loss


def build_synthetic_batch(config: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic signals that do not encode any industrial observation."""
    batch_size = 6
    length = int(config["data"]["sequence_length"])
    positions = torch.linspace(0.0, 4.0 * torch.pi, length)
    signals = []
    for sample_index in range(batch_size):
        views = []
        for view_index in range(int(config["data"]["num_views"])):
            phase = 0.12 * sample_index + 0.20 * view_index
            views.append(torch.sin(positions + phase) + 0.1 * torch.cos(2.0 * positions))
        signals.append(torch.stack(views))
    signal_batch = torch.stack(signals)
    signal_batch[1, 2, 18:26] = 0.0
    view_mask = torch.ones(batch_size, int(config["data"]["num_views"]))
    view_mask[4, 1] = 0.0
    signal_batch[4, 1] = 0.0
    labels = torch.arange(batch_size) % int(config["data"]["num_classes"])
    return signal_batch, labels, view_mask


def main() -> None:
    torch.manual_seed(7)
    config = load_demo_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTSGFM(config).to(device)
    model.train()
    signals, labels, view_mask = build_synthetic_batch(config)
    signals = signals.to(device)
    labels = labels.to(device)
    view_mask = view_mask.to(device)
    outputs = model(signals, view_mask)
    logits = outputs["logits"]
    assert logits is not None
    assert logits.shape == (signals.shape[0], int(config["data"]["num_classes"]))
    assert torch.isfinite(logits).all()
    loss, components = total_loss(outputs, labels, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(value >= 0.0 for value in components.values())
    print(f"device: {'GPU' if device.type == 'cuda' else 'CPU'}")
    print(f"output shape: {tuple(logits.shape)}")
    print("forward/loss/backward: passed")
    print("inputs: synthetic demonstration only")


if __name__ == "__main__":
    main()

