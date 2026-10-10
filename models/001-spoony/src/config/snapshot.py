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


def _json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported setting: {type(value).__name__}")


def normalize_settings(settings: dict[str, object]) -> dict:
    """Convert dataclasses/Paths to finite, JSON-compatible checkpoint values."""
    return json.loads(json.dumps(settings, default=_json_default, allow_nan=False))
