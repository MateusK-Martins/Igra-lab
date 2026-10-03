"""Typed, immutable configuration contracts for a decoder language model."""

from dataclasses import dataclass

from src.config.errors import ConfigError


@dataclass(frozen=True)
class TokenizerConfig:
    """Settings used to train one byte-level BPE tokenizer."""

    target_vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.target_vocab_size <= 0:
            raise ConfigError("target_vocab_size must be greater than zero")
        if self.min_frequency <= 0:
            raise ConfigError("min_frequency must be greater than zero")
        if not self.special_tokens:
            raise ConfigError("special_tokens must include an unknown token")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ConfigError("special_tokens must not contain duplicates")


@dataclass(frozen=True)
class ChunkingConfig:
    """Settings that turn a token stream into next-token training examples."""

    context_length: int
    stride: int

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ConfigError("context_length must be greater than zero")
        if self.stride <= 0:
            raise ConfigError("stride must be greater than zero")


@dataclass(frozen=True)
class DataConfig:
    """All settings required to prepare one pretraining dataset."""

    tokenizer: TokenizerConfig
    chunking: ChunkingConfig
