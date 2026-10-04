"""Tests for saved configuration snapshots."""

import json

import pytest

from src.config.snapshot import save_config_snapshot
from src.config.types import ChunkingConfig, DataConfig, TokenizerConfig


def test_saves_resolved_dataclass_configuration(tmp_path) -> None:
    path = tmp_path / "run" / "resolved-config.json"

    config = DataConfig(
        tokenizer=TokenizerConfig(4096, 2, ("<|unk|>", "<|eot|>")),
        chunking=ChunkingConfig(128, 128),
    )
    save_config_snapshot(config, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "chunking": {"context_length": 128, "stride": 128},
        "tokenizer": {
            "min_frequency": 2,
            "special_tokens": ["<|unk|>", "<|eot|>"],
            "target_vocab_size": 4096,
        },
    }


def test_rejects_non_dataclass_configuration(tmp_path) -> None:
    with pytest.raises(TypeError, match="dataclass instance"):
        save_config_snapshot({"tokenizer": "bpe"}, tmp_path / "config.json")
