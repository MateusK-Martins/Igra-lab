"""Typed, immutable configuration contracts for a decoder language model."""

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class PreparationConfig:
    tokenizer: TokenizerConfig
    output_directory: Path
    end_of_document_token: str

    def __post_init__(self) -> None:
        if self.end_of_document_token not in self.tokenizer.special_tokens:
            raise ConfigError("End-of-document token must be a tokenizer special token")


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool
    seed: int

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ConfigError("batch_size must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ConfigError("num_workers must be a non-negative integer")
        for field in ("shuffle", "pin_memory", "drop_last"):
            if type(getattr(self, field)) is not bool:
                raise ConfigError(f"{field} must be a boolean")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ConfigError("seed must be an integer between 0 and 2**64 - 1")
