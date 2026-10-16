import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheEntry, MatrixCacheEntry
from src.model.feedforward import LinearImpl, SwiGLUImpl
from src.model.latent_attention import LatentAttentionImpl
from src.model.position import PositionRotation
from src.model.stability import RMSNormImpl


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

    def cache(self, layer_id: int) -> CacheEntry | MatrixCacheEntry | None:
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
class GQAAttention(BlockDefinition):
    input_features: int
    output_features: int

    num_query_heads: int
    num_kv_heads: int
    head_features: int
    window_size: int
    dropout: float
    position_rotation: PositionRotation
    bias: bool = False
    residual: bool = False

    def __post_init__(self) -> None:
        _positive_integer(self.input_features, "input_features")
        _positive_integer(self.output_features, "output_features")
        _positive_integer(self.num_query_heads, "num_query_heads")
        _positive_integer(self.num_kv_heads, "num_kv_heads")
        _positive_integer(self.head_features, "head_features")

        if type(self.window_size) is not int or self.window_size < 0:
            raise ValueError("window_size must be a non-negative integer")

        if (
            type(self.dropout) not in (int, float)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1)")

        if type(self.bias) is not bool:
            raise ValueError("bias must be a boolean")

        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

    def build(self) -> nn.Module:
        return GQAAttentionImpl(
            self.input_features,
            self.output_features,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_features,
            self.window_size,
            self.dropout,
            self.bias,
            self.position_rotation,
        )

    def cache(self, layer_id: int) -> CacheEntry:
        shape = (
            0,
            self.num_kv_heads,
            0,
            self.head_features,
        )

        return CacheEntry(
            layer_id=f"GQAAttentionImpl-{layer_id}",
            bounded=False,
            keys=torch.empty(shape),
            values=torch.empty(shape),
            count=0,
            pointer=0,
        )


@dataclass(frozen=True)
class LatentAttention(BlockDefinition):
    input_features: int
    mode: str
    residual: bool = False
    rope_base: float | None = None
    matrix_features: int = 4

    @property
    def output_features(self):
        return self.input_features

    def __post_init__(self):
        _positive_integer(self.input_features, "input_features")
        _positive_integer(self.matrix_features, "matrix_features")
        if (
            self.mode.startswith("sma")
            and self.rope_base is not None
            and self.matrix_features % 2
        ):
            raise ValueError("matrix_features must be even for RoPE")
        if not self.mode.startswith("sma") and self.matrix_features != 4:
            raise ValueError("matrix_features only applies to matrix attention")
        if self.mode not in ("a16", "a4", "sma4", "sma4_key", "sma4_key_gelu16"):
            raise ValueError("mode must be a16, a4, sma4, sma4_key or sma4_key_gelu16")
        if self.rope_base is not None and (
            isinstance(self.rope_base, bool)
            or not isinstance(self.rope_base, (int, float))
            or not math.isfinite(self.rope_base)
            or self.rope_base <= 0
        ):
            raise ValueError("rope_base must be finite and positive or None")
        if type(self.residual) is not bool:
            raise ValueError("residual must be a boolean")

    def build(self):
        # Different projection sizes cannot shift initialization of the shared FFNs.
        with torch.random.fork_rng(devices=[]):
            return LatentAttentionImpl(
                self.input_features,
                self.mode,
                rope_base=self.rope_base,
                matrix_features=self.matrix_features,
            )

    def cache(self, layer_id):
        suffix = (
            f"-m{self.matrix_features}"
            if self.mode.startswith("sma") and self.matrix_features != 4
            else ""
        )
        name = f"LatentAttentionImpl{suffix}{'-rope' if self.rope_base is not None else ''}-{layer_id}"
        if self.mode in ("sma4", "sma4_key", "sma4_key_gelu16"):
            return MatrixCacheEntry(
                name,
                torch.empty(0, 0, self.matrix_features, self.matrix_features),
                keys=torch.empty(0, 0, self.matrix_features)
                if self.mode in ("sma4_key", "sma4_key_gelu16")
                else None,
            )
        features = 16 if self.mode == "a16" else 4
        shape = (0, 1, 0, features)
        return CacheEntry(name, False, torch.empty(shape), torch.empty(shape), 0, 0)
