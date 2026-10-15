import math
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn

from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheEntry, StateEntry
from src.model.cumulative import CumulativeMixerImpl
from src.model.feedforward import LinearImpl, SwiGLUImpl
from src.model.multikey import MultiKeyMixerImpl
from src.model.no_k_variant import NoKVariantImpl
from src.model.position import PositionRotation
from src.model.stability import RMSNormImpl
from src.model.state_mechanisms import MassNormalizedImpl, SignedStateImpl


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

    def cache(self, layer_id: int) -> CacheEntry | StateEntry | None:
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
class CumulativeMixer(BlockDefinition):
    input_features: int
    eps: float = 1e-6
    residual: bool = False
    bias: bool = False
    normalization: Literal["rms", "soft_rms"] = "rms"
    key_decay: bool = False
    value_decay: bool = False
    decay_init: float = 0.99
    decay_mode: Literal["diagonal", "token"] = "diagonal"

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
        if self.normalization not in ("rms", "soft_rms"):
            raise ValueError("normalization must be rms or soft_rms")
        if (
            type(self.decay_init) not in (int, float)
            or not math.isfinite(self.decay_init)
            or not 0 < self.decay_init < 1
        ):
            raise ValueError(
                "decay_init must be finite and strictly between zero and one"
            )
        if self.decay_mode not in ("diagonal", "token"):
            raise ValueError("decay_mode must be diagonal or token")
        for name in ("residual", "bias", "key_decay", "value_decay"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def build(self) -> nn.Module:
        return CumulativeMixerImpl(
            self.input_features,
            self.eps,
            bias=self.bias,
            normalization=self.normalization,
            key_decay=self.key_decay,
            value_decay=self.value_decay,
            decay_init=self.decay_init,
            decay_mode=self.decay_mode,
        )

    def cache(self, layer_id: int) -> StateEntry:
        return StateEntry(
            f"CumulativeMixerImpl-{layer_id}",
            torch.empty(0, 1, self.input_features),
            torch.empty(0, 1, self.input_features),
        )


@dataclass
class SignedState(BlockDefinition):
    input_features: int
    token_decay: bool = True
    eps: float = 1e-6
    decay_init: float = 0.99
    residual: bool = False

    @property
    def output_features(self):
        return self.input_features

    def __post_init__(self):
        CumulativeMixer(
            self.input_features,
            eps=self.eps,
            decay_init=self.decay_init,
            residual=self.residual,
        )
        if type(self.token_decay) is not bool:
            raise ValueError("token_decay must be boolean")

    def build(self):
        return SignedStateImpl(
            self.input_features, self.eps, self.decay_init, self.token_decay
        )

    def cache(self, layer_id: int):
        return StateEntry(
            f"SignedStateImpl-{layer_id}",
            torch.empty(0, 1, 0),
            torch.empty(0, 1, self.input_features),
        )


@dataclass
class MassNormalized(BlockDefinition):
    input_features: int
    eps: float = 1e-6
    decay_init: float = 0.99
    residual: bool = False

    @property
    def output_features(self):
        return self.input_features

    def __post_init__(self):
        CumulativeMixer(
            self.input_features,
            eps=self.eps,
            decay_init=self.decay_init,
            residual=self.residual,
        )

    def build(self):
        return MassNormalizedImpl(self.input_features, self.eps, self.decay_init)

    def cache(self, layer_id: int):
        return StateEntry(
            f"MassNormalizedImpl-{layer_id}",
            torch.empty(0, 1, 2 * self.input_features),
            torch.empty(0, 1, 2 * self.input_features),
        )


@dataclass
class MultiKeyMixer(BlockDefinition):
    input_features: int
    num_keys: int = 4
    tied_query: bool = True
    eps: float = 1e-6
    decay_init: float = 0.99
    residual: bool = False

    @property
    def output_features(self) -> int:
        return self.input_features

    def __post_init__(self):
        _positive_integer(self.input_features, "input_features")
        _positive_integer(self.num_keys, "num_keys")
        for name in ("tied_query", "residual"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            type(self.eps) not in (int, float)
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError("eps must be finite and positive")
        if (
            type(self.decay_init) not in (int, float)
            or not math.isfinite(self.decay_init)
            or not 0 < self.decay_init < 1
        ):
            raise ValueError("decay_init must be strictly between zero and one")

    def build(self):
        return MultiKeyMixerImpl(
            self.input_features,
            self.num_keys,
            self.tied_query,
            self.eps,
            self.decay_init,
        )

    def cache(self, layer_id: int) -> StateEntry:
        return StateEntry(
            f"MultiKeyMixerImpl-{layer_id}",
            torch.empty(0, 1, self.num_keys * self.input_features),
            torch.empty(0, 1, self.input_features),
        )


@dataclass
class NoKVariant(BlockDefinition):
    input_features: int
    mode: Literal["write_gate", "signed_readout", "lowrank"] = "write_gate"
    rank: int = 32
    num_heads: int = 1
    retention_initials: tuple[float, ...] | None = None
    write_initials: tuple[float, ...] | None = None
    eps: float = 1e-6
    decay_init: float = 0.99
    residual: bool = False

    @property
    def output_features(self):
        return self.input_features

    def __post_init__(self):
        CumulativeMixer(
            self.input_features,
            eps=self.eps,
            decay_init=self.decay_init,
            residual=self.residual,
        )
        if self.mode not in ("write_gate", "signed_readout", "lowrank"):
            raise ValueError("mode must be write_gate, signed_readout or lowrank")
        _positive_integer(self.rank, "rank")
        if self.mode == "lowrank" and self.rank > self.input_features:
            raise ValueError("rank must not exceed feature width")

        _positive_integer(self.num_heads, "num_heads")
        if self.input_features % self.num_heads:
            raise ValueError("feature width must be divisible by num_heads")
        if self.mode != "write_gate" and (
            self.num_heads != 1
            or self.retention_initials is not None
            or self.write_initials is not None
        ):
            raise ValueError("head and gate preferences require write_gate mode")
        for values in (self.retention_initials, self.write_initials):
            if values is not None:
                if not isinstance(values, tuple) or len(values) != self.num_heads:
                    raise ValueError(
                        "gate preferences must contain one probability per head"
                    )
                if any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not 0 < v < 1
                    for v in values
                ):
                    raise ValueError(
                        "gate probabilities must be strictly between zero and one"
                    )

    def build(self):
        return NoKVariantImpl(
            self.input_features,
            self.eps,
            self.decay_init,
            self.mode,
            self.rank,
            self.num_heads,
            self.retention_initials,
            self.write_initials,
        )

    def cache(self, layer_id: int):
        suffix = f"-heads{self.num_heads}" if self.num_heads > 1 else ""
        return StateEntry(
            f"NoKVariantImpl-{self.mode}{suffix}-{layer_id}",
            torch.empty(0, 1, 0),
            torch.empty(0, 1, self.input_features),
        )
