"""Tests for typed model configuration contracts."""

import pytest

from src.config.errors import ConfigError
from src.config.types import ChunkingConfig, DataLoaderConfig, TokenizerConfig


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("num_workers", -1),
        ("num_workers", True),
        ("shuffle", "true"),
        ("pin_memory", 1),
        ("drop_last", None),
        ("seed", -1),
        ("seed", 2**64),
        ("seed", True),
    ],
)
def test_loader_config_rejects_invalid_settings(field, value) -> None:
    settings = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
        "seed": 42,
    }
    settings[field] = value
    with pytest.raises(ConfigError, match=field):
        DataLoaderConfig(**settings)


def test_loader_config_accepts_zero_workers_and_seed_boundaries() -> None:
    for seed in (0, 2**64 - 1):
        config = DataLoaderConfig(1, False, 0, False, False, seed)
        assert config.num_workers == 0
