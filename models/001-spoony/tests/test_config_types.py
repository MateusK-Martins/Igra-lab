"""Tests for typed model configuration contracts."""

import pytest

from src.config.errors import ConfigError
from src.config.types import ChunkingConfig, TokenizerConfig


def test_tokenizer_config_accepts_valid_values() -> None:
    config = TokenizerConfig(
        target_vocab_size=4096,
        min_frequency=2,
        special_tokens=("<|unk|>", "<|eot|>"),
    )

    assert config.target_vocab_size == 4096


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_vocab_size", 0, "target_vocab_size"),
        ("min_frequency", 0, "min_frequency"),
    ],
)
def test_tokenizer_config_rejects_invalid_number(
    field: str,
    value: int,
    message: str,
) -> None:
    values = {
        "target_vocab_size": 4096,
        "min_frequency": 2,
        "special_tokens": ("<|unk|>", "<|eot|>"),
    }
    values[field] = value

    with pytest.raises(ConfigError, match=message):
        TokenizerConfig(**values)


def test_tokenizer_config_rejects_empty_or_duplicate_special_tokens() -> None:
    with pytest.raises(ConfigError, match="must include"):
        TokenizerConfig(target_vocab_size=4096, min_frequency=2, special_tokens=())

    with pytest.raises(ConfigError, match="duplicates"):
        TokenizerConfig(
            target_vocab_size=4096,
            min_frequency=2,
            special_tokens=("<|unk|>", "<|unk|>"),
        )


@pytest.mark.parametrize(
    ("context_length", "stride"),
    [(0, 1), (1, 0)],
)
def test_chunking_config_rejects_non_positive_values(
    context_length: int,
    stride: int,
) -> None:
    with pytest.raises(ConfigError):
        ChunkingConfig(context_length=context_length, stride=stride)
