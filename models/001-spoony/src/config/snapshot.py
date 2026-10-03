"""Serialize resolved dataclass configurations as run artifacts."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def save_config_snapshot(config: object, path: Path) -> None:
    """Write one resolved dataclass configuration as readable JSON."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("config must be a dataclass instance")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
