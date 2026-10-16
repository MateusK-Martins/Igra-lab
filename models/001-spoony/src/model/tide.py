"""Signed selective-write vector memory with global RMS normalization."""

import math

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.decay import TokenDecayImpl
from src.model.scan import affine_prefix_sum
from src.model.stability import RMSNormImpl


class TideMixerImpl(nn.Module):
    def __init__(
        self, features: int, eps: float, retention_initials=None, write_initials=None
    ):
        super().__init__()
        self.features = features
        self.q_proj = nn.Linear(features, features, bias=False)
        # Preserve Tide's historical seeded initialization without retaining
        # the unused projection allocated by its former experimental parent.
        nn.init.kaiming_uniform_(torch.empty(features, features), a=math.sqrt(5))
        self.v_proj = nn.Linear(features, features, bias=False)
        self.value_norm = RMSNormImpl(features, eps)
        self.value_decay_gate = TokenDecayImpl(features, 0.99)
        self.write_gate = TokenDecayImpl(features, 0.5)
        denominator = max(features - 1, 1)
        retention = (
            retention_initials
            if retention_initials is not None
            else tuple(0.90 + 0.095 * j / denominator for j in range(features))
        )
        writing = (
            write_initials
            if write_initials is not None
            else tuple(0.20 - 0.18 * j / denominator for j in range(features))
        )
        with torch.no_grad():
            self.value_decay_gate.bias.copy_(
                torch.logit(self.value_decay_gate.bias.new_tensor(retention))
            )
            self.write_gate.bias.copy_(
                torch.logit(self.write_gate.bias.new_tensor(writing))
            )
        self.layer_id = ""

    def assign_id(self, layer_id: int):
        self.layer_id = f"TideMixerImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Tide cache requires no_grad")
        q = self.q_proj(x).sigmoid()
        u = self.v_proj(x).tanh()
        dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
        u = self.write_gate(x).to(dtype) * u.to(dtype)
        initial = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.value_sum.shape != (x.shape[0], 1, self.features)
                    or previous.key_sum.shape != (x.shape[0], 1, 0)
                    or previous.value_sum.dtype != dtype
                ):
                    raise ValueError("Tide cache dimensions or dtype changed")
                initial = previous.value_sum
        state = affine_prefix_sum(u, self.value_decay_gate(x).to(dtype), initial)
        read = self.value_norm(state)
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id,
                state.new_empty(x.shape[0], 1, 0),
                state[:, -1:],
                x.shape[1],
            )
        return q * read.to(q.dtype)
