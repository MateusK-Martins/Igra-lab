"""Causal feature-wise cumulative mixing, with two constant-size running sums."""

import math
from typing import Literal

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.scan import decayed_prefix_sum
from src.model.stability import RMSNormImpl, SoftRMSNormImpl


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
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=bias)
        self.k_proj = nn.Linear(features, features, bias=bias)
        self.v_proj = nn.Linear(features, features, bias=bias)
        self.normalization = normalization
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
        for name, enabled in (
            ("key_decay_logits", key_decay),
            ("value_decay_logits", value_decay),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.full((features,), math.log(decay_init / (1 - decay_init)))
                )
                if enabled
                else None,
            )
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
        key_sum = decayed_prefix_sum(k.to(working_dtype), key_decay, initial_key)
        normalized_keys = self.key_norm(key_sum)
        value_sum = decayed_prefix_sum(
            v.to(working_dtype) * normalized_keys, value_decay, initial_value
        )
        result = q * self.value_norm(value_sum).to(q.dtype)
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id, key_sum[:, -1:], value_sum[:, -1:], x.shape[1]
            )
        # build_block supplies x + result when the definition declares residual=True.
        return result
