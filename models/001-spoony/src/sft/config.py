"""Configuration contracts for supervised instruction tuning."""

from dataclasses import dataclass
from pathlib import Path

from src.config.errors import ConfigError


@dataclass(frozen=True)
class SFTDataConfig:
    train_path: Path
    validation_path: Path
    tokenizer_path: Path
    context_length: int
    instruction_prefix: str = "User: "
    response_prefix: str = "\nAssistant: "
    end_token: str = "<|eot|>"

    def __post_init__(self) -> None:
        for name in ("train_path", "validation_path", "tokenizer_path"):
            if not isinstance(getattr(self, name), Path):
                raise ConfigError(f"{name} must be a Path")
        if type(self.context_length) is not int or self.context_length <= 0:
            raise ConfigError("context_length must be a positive integer")
        for name in ("instruction_prefix", "response_prefix", "end_token"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ConfigError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SFTInitializationConfig:
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, Path):
            raise ConfigError("checkpoint_path must be a Path")
