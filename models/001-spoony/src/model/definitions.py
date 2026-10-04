from dataclasses import dataclass
from typing import Protocol

from torch import nn

from src.model.composition import build_block
from src.model.feedforward import SwiGLUFeedForward


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class EmbeddingDefinition(Protocol):
    @property
    def output_features(self) -> int: ...

    def build(self, *, vocab_size: int) -> nn.Embedding: ...


class BlockDefinition(Protocol):
    @property
    def input_features(self) -> int: ...

    @property
    def output_features(self) -> int: ...

    @property
    def residual(self) -> bool: ...

    def build(self) -> nn.Module: ...


@dataclass(frozen=True)
class Linear:
    input_features: int
    output_features: int
    residual: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")
        _positive_integer(self.output_features, "output_features")
        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    def build(self) -> nn.Module:
        return nn.Linear(
            self.input_features,
            self.output_features,
        )


@dataclass(frozen=True)
class Repeat:
    times: int
    blocks: list[BlockDefinition]
    residual: bool = False

    @property
    def input_features(self) -> int:
        return self.blocks[0].input_features

    @property
    def output_features(self) -> int:
        return self.blocks[-1].output_features

    def build(self) -> nn.Module:
        modules = []

        for _ in range(self.times):
            for block in self.blocks:
                modules.append(build_block(block))

        return nn.Sequential(*modules)


@dataclass(frozen=True)
class Sequential:
    blocks: list[BlockDefinition]
    residual: bool = False

    @property
    def input_features(self) -> int:
        return self.blocks[0].input_features

    @property
    def output_features(self) -> int:
        return self.blocks[-1].output_features

    def build(self) -> nn.Module:
        return nn.Sequential(*(build_block(block) for block in self.blocks))


@dataclass(frozen=True)
class SwiGLU:
    input_features: int
    hidden_features: int
    output_features: int
    residual: bool = False
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("input_features", "hidden_features", "output_features"):
            _positive_integer(getattr(self, name), name)
        for name in ("residual", "bias"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def build(self) -> nn.Module:
        return SwiGLUFeedForward(
            self.input_features,
            self.hidden_features,
            self.output_features,
            bias=self.bias,
        )


@dataclass(frozen=True)
class Embedding:
    output_features: int

    def __post_init__(self) -> None:
        _positive_integer(self.output_features, "output_features")

    def build(self, *, vocab_size: int) -> nn.Embedding:
        _positive_integer(vocab_size, "vocab_size")
        return nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.output_features,
        )


class LMHeadDefinition(Protocol):
    @property
    def input_features(self) -> int: ...

    @property
    def tie_to_embedding(self) -> bool: ...

    def build(
        self,
        *,
        vocab_size: int,
        embedding: nn.Embedding,
    ) -> nn.Linear: ...


@dataclass(frozen=True)
class LMHead:
    input_features: int
    tie_to_embedding: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")
        if type(self.tie_to_embedding) is not bool:
            raise ValueError("tie_to_embedding must be a boolean")

    def build(self, *, vocab_size: int, embedding: nn.Embedding) -> nn.Linear:
        _positive_integer(vocab_size, "vocab_size")
        if self.tie_to_embedding:
            if embedding.num_embeddings != vocab_size:
                raise ValueError(
                    "Embedding vocabulary size must match LM head vocabulary"
                )
            if embedding.embedding_dim != self.input_features:
                raise ValueError(
                    "Tied LM head input_features must match embedding width"
                )

        head = nn.Linear(self.input_features, vocab_size, bias=False)
        if self.tie_to_embedding:
            head.weight = embedding.weight
        return head
