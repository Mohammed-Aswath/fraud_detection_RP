"""Central configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return config
