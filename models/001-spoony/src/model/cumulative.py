"""Causal feature-wise cumulative mixing, with two constant-size running sums."""

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.stability import RMSNormImpl


class CumulativeMixerImpl(nn.Module):
    def __init__(self, features: int, eps: float, *, bias: bool) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=bias)
        self.k_proj = nn.Linear(features, features, bias=bias)
        self.v_proj = nn.Linear(features, features, bias=bias)
        self.key_norm = RMSNormImpl(features, eps)
        self.value_norm = RMSNormImpl(features, eps)
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
        key_sum = k.to(working_dtype).cumsum(dim=1)
        previous = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.key_sum.shape != (x.shape[0], 1, x.shape[2])
                    or previous.key_sum.dtype != working_dtype
                ):
                    raise ValueError("Recurrent cache batch, features or dtype changed")
                key_sum = key_sum + previous.key_sum
        normalized_keys = self.key_norm(key_sum)
        value_sum = (v.to(working_dtype) * normalized_keys).cumsum(dim=1)
        if previous is not None and previous.count:
            value_sum = value_sum + previous.value_sum
        result = q * self.value_norm(value_sum).to(q.dtype)
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id, key_sum[:, -1:], value_sum[:, -1:], x.shape[1]
            )
        # build_block supplies x + result when the definition declares residual=True.
        return result
