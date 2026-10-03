"""Tests for environment-variable resolution in configuration values."""

import pytest

from src.config.environment import resolve_environment_values
from src.config.errors import ConfigError


def test_resolves_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOONY_DATA_ROOT", "/tmp/spoony-data")

    resolved = resolve_environment_values("${SPOONY_DATA_ROOT}")

    assert resolved == "/tmp/spoony-data"


def test_resolves_environment_variable_inside_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPOONY_DATA_ROOT", "/tmp/spoony-data")

    resolved = resolve_environment_values("${SPOONY_DATA_ROOT}/raw/tinystories")

    assert resolved == "/tmp/spoony-data/raw/tinystories"


def test_preserves_values_without_placeholders() -> None:
    assert resolve_environment_values("byte-level-bpe") == "byte-level-bpe"
    assert resolve_environment_values(4096) == 4096
    assert resolve_environment_values(True) is True


def test_resolves_nested_dictionaries_and_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPOONY_DATA_ROOT", "/tmp/spoony-data")
    raw_config = {
        "paths": {"root": "${SPOONY_DATA_ROOT}"},
        "files": ["${SPOONY_DATA_ROOT}/train.txt", "validation.txt"],
    }

    resolved = resolve_environment_values(raw_config)

    assert resolved == {
        "paths": {"root": "/tmp/spoony-data"},
        "files": ["/tmp/spoony-data/train.txt", "validation.txt"],
    }
    assert raw_config["paths"]["root"] == "${SPOONY_DATA_ROOT}"


def test_rejects_missing_environment_variable() -> None:
    with pytest.raises(ConfigError, match="missing environment variable"):
        resolve_environment_values("${SPOONY_DATA_ROOT}")
