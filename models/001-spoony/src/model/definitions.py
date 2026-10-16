import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from src.model.cache_storage import StateEntry
from src.model.feedforward import LinearImpl, SwiGLUImpl
from src.model.stability import RMSNormImpl
from src.model.tide import TideMixerImpl


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class EmbeddingDefinition(Protocol):
    output_features: int

    def build(self, *, vocab_size: int) -> nn.Embedding: ...


class BlockDefinition(Protocol):
    input_features: int
    output_features: int
    residual: bool

    def build(self) -> nn.Module: ...

    def cache(self, layer_id: int) -> StateEntry | None:
        return None

    def unpack(self) -> list["BlockDefinition"]:
        return [self]


@dataclass(frozen=True)
class Linear(BlockDefinition):
    input_features: int
    output_features: int
    residual: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")
        _positive_integer(self.output_features, "output_features")

        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    def build(self) -> nn.Module:
        return LinearImpl(
            self.input_features,
            self.output_features,
        )


@dataclass(frozen=True)
class RMSNorm(BlockDefinition):
    input_features: int
    eps: float
    residual: bool = False

    @property
    def output_features(self) -> int:
        return self.input_features

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")

        if (
            type(self.eps) not in (int, float)
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError("eps must be finite and greater than zero")

        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    def build(self) -> nn.Module:
        return RMSNormImpl(
            self.input_features,
            self.eps,
        )


@dataclass(frozen=True)
class Repeat(BlockDefinition):
    times: int
    blocks: list[BlockDefinition]
    residual: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.times, "times")

        if not self.blocks:
            raise ValueError("blocks must not be empty")

        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    @property
    def input_features(self) -> int:
        return self.blocks[0].input_features

    @property
    def output_features(self) -> int:
        return self.blocks[-1].output_features

    def build(self) -> nn.Module:
        raise TypeError("Repeat must be unpacked before building")

    def unpack(self) -> list[BlockDefinition]:
        self.__post_init__()
        definitions: list[BlockDefinition] = []

        for _ in range(self.times):
            for block in self.blocks:
                definitions.extend(block.unpack())

        return definitions


@dataclass(frozen=True)
class Sequential(BlockDefinition):
    blocks: list[BlockDefinition]
    residual: bool = False

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("blocks must not be empty")

        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    @property
    def input_features(self) -> int:
        return self.blocks[0].input_features

    @property
    def output_features(self) -> int:
        return self.blocks[-1].output_features

    def build(self) -> nn.Module:
        raise TypeError("Sequential must be unpacked before building")

    def unpack(self) -> list[BlockDefinition]:
        self.__post_init__()
        definitions: list[BlockDefinition] = []

        for block in self.blocks:
            definitions.extend(block.unpack())

        return definitions


@dataclass(frozen=True)
class SwiGLU(BlockDefinition):
    input_features: int
    hidden_features: int
    output_features: int
    residual: bool = False
    bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "input_features",
            "hidden_features",
            "output_features",
        ):
            _positive_integer(getattr(self, name), name)

        for name in ("residual", "bias"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def build(self) -> nn.Module:
        return SwiGLUImpl(
            self.input_features,
            self.hidden_features,
            self.output_features,
            bias=self.bias,
        )


@dataclass(frozen=True)
class Embedding(EmbeddingDefinition):
    output_features: int
    init_std: float = 1.0

    def __post_init__(self) -> None:
        _positive_integer(self.output_features, "output_features")
        if (
            type(self.init_std) not in (int, float)
            or not math.isfinite(self.init_std)
            or self.init_std <= 0
        ):
            raise ValueError("init_std must be a finite positive number")

    def build(self, *, vocab_size: int) -> nn.Embedding:
        _positive_integer(vocab_size, "vocab_size")

        embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.output_features,
        )
        # nn.Embedding starts with N(0, 1). Scaling preserves the same random
        # draw and RNG state, including the initialization of subsequent blocks.
        with torch.no_grad():
            embedding.weight.mul_(self.init_std)
        return embedding


class LMHeadDefinition(Protocol):
    input_features: int
    tie_to_embedding: bool

    def build(
        self,
        *,
        vocab_size: int,
        embedding: nn.Embedding,
    ) -> nn.Linear: ...


@dataclass(frozen=True)
class LMHead(LMHeadDefinition):
    input_features: int
    tie_to_embedding: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")

        if type(self.tie_to_embedding) is not bool:
            raise ValueError("tie_to_embedding must be a boolean")

    def build(
        self,
        *,
        vocab_size: int,
        embedding: nn.Embedding,
    ) -> nn.Linear:
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

        head = nn.Linear(
            self.input_features,
            vocab_size,
            bias=False,
        )

        if self.tie_to_embedding:
            head.weight = embedding.weight

        return head


@dataclass(frozen=True)
class TideMixer(BlockDefinition):
    input_features: int
    eps: float = 1e-6
    retention_initials: tuple[float, ...] | None = None
    write_initials: tuple[float, ...] | None = None
    residual: bool = False

    @property
    def output_features(self):
        return self.input_features

    def __post_init__(self):
        RMSNorm(self.input_features, self.eps, self.residual)
        for values in (self.retention_initials, self.write_initials):
            if values is not None:
                if not isinstance(values, tuple) or len(values) != self.input_features:
                    raise ValueError(
                        "gate preferences require one probability per feature"
                    )
                if any(type(v) not in (int, float) or not 0 < v < 1 for v in values):
                    raise ValueError(
                        "gate probabilities must be strictly between zero and one"
                    )

    def build(self):
        return TideMixerImpl(
            self.input_features, self.eps, self.retention_initials, self.write_initials
        )

    def cache(self, layer_id: int):
        return StateEntry(
            f"TideMixerImpl-{layer_id}",
            torch.empty(0, 1, 0),
            torch.empty(0, 1, self.input_features),
        )
