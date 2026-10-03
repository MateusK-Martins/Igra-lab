from pathlib import Path

import pytest

from src.config.errors import ConfigError
from src.config.paths import config_path, find_project_root


def test_finds_model_root_from_nested_directory() -> None:
    nested_directory = Path(__file__).parent

    root = find_project_root(nested_directory)

    assert (root / "pyproject.toml").is_file()
    assert (root / "src").is_dir()


def test_resolves_existing_tokenizer_config() -> None:
    path = config_path("tokenizer/bpe-4k.toml")

    assert path.is_file()
    assert path.name == "bpe-4k.toml"


def test_rejects_path_outside_project() -> None:
    with pytest.raises(ConfigError, match="escapes"):
        config_path("../../outside-model")


def test_fails_when_no_project_root_exists(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not find"):
        find_project_root(tmp_path)
