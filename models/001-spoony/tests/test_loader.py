from pathlib import Path

import pytest

from src.config.errors import ConfigError
from src.config.loader import load_toml


def test_loads_valid_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        '[tokenizer]\nkind = "byte-level-bpe"\ntarget_vocab_size = 4096\n',
        encoding="utf-8",
    )

    config = load_toml(config_path)

    assert config == {
        "tokenizer": {
            "kind": "byte-level-bpe",
            "target_vocab_size": 4096,
        }
    }


def test_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"

    with pytest.raises(ConfigError, match="does not exist"):
        load_toml(missing_path)


def test_rejects_directory(tmp_path: Path) -> None:
    directory_path = tmp_path / "configs"
    directory_path.mkdir()

    with pytest.raises(ConfigError, match="not a file"):
        load_toml(directory_path)


def test_rejects_invalid_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[tokenizer\nkind = ", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_toml(config_path)
