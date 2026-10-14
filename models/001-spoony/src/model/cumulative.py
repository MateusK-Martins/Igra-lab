"""Causal feature-wise cumulative mixing, with two constant-size running sums."""

import math
from typing import Literal

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.decay import TokenDecayImpl
from src.model.scan import affine_prefix_sum, decayed_prefix_sum
from src.model.stability import HeadRMSNormImpl, RMSNormImpl, SoftRMSNormImpl


class CumulativeMixerImpl(nn.Module):
    def __init__(
        self,
        features: int,
        eps: float,
        *,
        bias: bool,
        normalization: Literal["rms", "soft_rms"] = "rms",
        key_decay: bool = False,
        value_decay: bool = False,
        decay_init: float = 0.99,
        decay_mode: Literal["diagonal", "token"] = "diagonal",
        normalization_scope: str = "global",
        readout: str = "elementwise",
        num_heads: int = 1,
        query_head_features: int = 48,
        head_mlp_hidden: int = 24,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=bias)
        self.readout = readout
        self.num_heads = num_heads
        self.query_head_features = query_head_features
        # Draw the original full Q first; replace it inside an RNG fork so all
        # unchanged parameters retain their original seed-42 initial values.
        if readout == "matrix":
            with torch.random.fork_rng(devices=[]):
                self.q_proj = nn.Linear(
                    features, num_heads * query_head_features, bias=False
                )
        self.k_proj = nn.Linear(features, features, bias=bias)
        self.v_proj = nn.Linear(features, features, bias=bias)
        self.normalization = normalization
        self.decay_mode = decay_mode
        self.key_norm = (
            RMSNormImpl(features, eps)
            if normalization == "rms"
            else SoftRMSNormImpl(features)
        )
        self.value_norm = (
            RMSNormImpl(features, eps)
            if normalization == "rms"
            else SoftRMSNormImpl(features)
        )
        if normalization_scope == "head":
            self.key_norm = HeadRMSNormImpl(features, eps, num_heads)
            self.value_norm = HeadRMSNormImpl(features, eps, num_heads)
        for name, enabled in (
            ("key_decay_logits", key_decay),
            ("value_decay_logits", value_decay),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.full((features,), math.log(decay_init / (1 - decay_init)))
                )
                if enabled and decay_mode == "diagonal"
                else None,
            )
        self.key_decay_gate = (
            TokenDecayImpl(features, decay_init)
            if key_decay and decay_mode == "token"
            else None
        )
        self.value_decay_gate = (
            TokenDecayImpl(features, decay_init)
            if value_decay and decay_mode == "token"
            else None
        )
        self.head_mlp = None
        self.register_parameter("pair_p", None)
        self.register_parameter("pair_w", None)
        with torch.random.fork_rng(devices=[]):
            if readout == "matrix":
                width = features // num_heads
                self.pair_p = nn.Parameter(
                    torch.empty(num_heads, width, query_head_features)
                )
                self.pair_w = nn.Parameter(
                    torch.empty(num_heads, query_head_features, width)
                )
                nn.init.normal_(self.pair_p, std=1 / math.sqrt(width))
                nn.init.normal_(self.pair_w, std=1 / math.sqrt(query_head_features))
            elif readout == "head_mlp":
                self.head_mlp = nn.Sequential(
                    nn.Linear(num_heads, head_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(head_mlp_hidden, num_heads),
                )
                nn.init.zeros_(self.head_mlp[-1].weight)
                nn.init.zeros_(self.head_mlp[-1].bias)
        self.layer_id = ""

    def assign_id(self, layer_id: int) -> None:
        self.layer_id = f"CumulativeMixerImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Cumulative cached execution requires no_grad")
        q = torch.sigmoid(self.q_proj(x))
        k = torch.tanh(self.k_proj(x))
        v = torch.sigmoid(self.v_proj(x))
        # Accumulate low-precision projections in FP32, preserving FP64 references.
        working_dtype = (
            torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        )
        previous = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count and (
                previous.key_sum.shape != (x.shape[0], 1, x.shape[2])
                or previous.key_sum.dtype != working_dtype
            ):
                raise ValueError("Recurrent cache batch, features or dtype changed")
        initial_key = (
            previous.key_sum if previous is not None and previous.count else None
        )
        initial_value = (
            previous.value_sum if previous is not None and previous.count else None
        )
        key_decay = (
            self.key_decay_logits.to(working_dtype).sigmoid()
            if self.key_decay_logits is not None
            else None
        )
        value_decay = (
            self.value_decay_logits.to(working_dtype).sigmoid()
            if self.value_decay_logits is not None
            else None
        )
        if self.key_decay_gate is not None:
            key_sum = affine_prefix_sum(
                k.to(working_dtype),
                self.key_decay_gate(x).to(working_dtype),
                initial_key,
            )
        else:
            key_sum = decayed_prefix_sum(k.to(working_dtype), key_decay, initial_key)
        normalized_keys = self.key_norm(key_sum)
        value_update = v.to(working_dtype) * normalized_keys
        if self.value_decay_gate is not None:
            value_sum = affine_prefix_sum(
                value_update, self.value_decay_gate(x).to(working_dtype), initial_value
            )
        else:
            value_sum = decayed_prefix_sum(value_update, value_decay, initial_value)
        result = self.read_values(q, self.value_norm(value_sum).to(q.dtype))
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id, key_sum[:, -1:], value_sum[:, -1:], x.shape[1]
            )
        # build_block supplies x + result when the definition declares residual=True.
        return result

    def read_values(self, q: Tensor, values: Tensor) -> Tensor:
        if self.readout == "elementwise":
            return q * values
        heads = values.reshape(*values.shape[:-1], self.num_heads, -1)
        if self.readout == "head_mlp":
            h = heads.transpose(-1, -2)
            mixed = (h + self.head_mlp(h)).transpose(-1, -2).reshape_as(values)
            return q * mixed
        queries = q.reshape(*q.shape[:-1], self.num_heads, self.query_head_features)
        u = torch.einsum("...hd,hdr->...hr", heads, self.pair_p)
        transformed = torch.einsum("...hr,hrd->...hd", queries * u, self.pair_w)
        return (heads.roll(-1, dims=-2) * transformed).reshape_as(values)
