"""Public demo configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_demo_config(path: Path | None = None) -> dict[str, Any]:
    """Load the non-sensitive configuration used by the demo and tests."""
    config_path = path or PROJECT_ROOT / "configs" / "demo.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required_sections = {"data", "model", "training"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config

