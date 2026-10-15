"""Signed vector states and signed-mass normalized cascaded recurrence."""

import math

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.cumulative import CumulativeMixerImpl
from src.model.scan import affine_prefix_sum, decayed_prefix_sum


class SignedStateImpl(CumulativeMixerImpl):
    def __init__(
        self,
        features: int,
        eps: float,
        decay_init: float,
        token_decay: bool,
        normalize_state: bool = True,
    ):
        super().__init__(
            features,
            eps,
            bias=False,
            value_decay=True,
            decay_mode="token",
            decay_init=decay_init,
        )
        self.features, self.token_decay = features, token_decay
        self.normalize_state = normalize_state
        if not normalize_state:
            del self.value_norm
        # Draw the original projections/gate first, preserving unchanged RNG values.
        del self.k_proj, self.key_norm
        if not token_decay:
            self.value_decay_gate = None
            self.value_decay_logits = nn.Parameter(
                torch.full((features,), math.log(decay_init / (1 - decay_init)))
            )

    def assign_id(self, layer_id: int):
        self.layer_id = f"SignedStateImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Signed-state cache requires no_grad")
        q, update = self.q_proj(x).sigmoid(), self.v_proj(x).tanh()
        dtype = (
            torch.float32
            if update.dtype in (torch.float16, torch.bfloat16)
            else update.dtype
        )
        initial = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.value_sum.shape != (x.shape[0], 1, self.features)
                    or previous.value_sum.dtype != dtype
                    or previous.key_sum.shape != (x.shape[0], 1, 0)
                ):
                    raise ValueError("Signed-state cache dimensions or dtype changed")
                initial = previous.value_sum
        if self.token_decay:
            state = affine_prefix_sum(
                update.to(dtype), self.value_decay_gate(x).to(dtype), initial
            )
        else:
            state = decayed_prefix_sum(
                update.to(dtype), self.value_decay_logits.to(dtype).sigmoid(), initial
            )
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id,
                state.new_empty(x.shape[0], 1, 0),
                state[:, -1:],
                x.shape[1],
            )
        read = self.value_norm(state) if self.normalize_state else state
        return q * read.to(q.dtype)


class MassNormalizedImpl(CumulativeMixerImpl):
    def __init__(self, features: int, eps: float, decay_init: float):
        super().__init__(
            features,
            eps,
            bias=False,
            value_decay=True,
            decay_mode="token",
            decay_init=decay_init,
        )
        self.features, self.eps = features, eps
        del self.key_norm, self.value_norm

    def assign_id(self, layer_id: int):
        self.layer_id = f"MassNormalizedImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Mass-normalized cache requires no_grad")
        q, k, v = (
            self.q_proj(x).sigmoid(),
            self.k_proj(x).tanh(),
            self.v_proj(x).sigmoid(),
        )
        dtype = torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        k, v = k.to(dtype), v.to(dtype)
        initial_k = initial_v = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.key_sum.shape != (x.shape[0], 1, 2 * self.features)
                    or previous.value_sum.shape != previous.key_sum.shape
                    or previous.key_sum.dtype != dtype
                ):
                    raise ValueError(
                        "Mass-normalized cache dimensions or dtype changed"
                    )
                initial_k, initial_v = previous.key_sum, previous.value_sum
        keys = decayed_prefix_sum(torch.cat((k, k.abs()), -1), initial=initial_k)
        key_num, key_den = keys.chunk(2, -1)
        normalized_key = key_num / key_den.clamp_min(self.eps)
        gate = self.value_decay_gate(x).to(dtype)
        # Denominator tracks positive write mass with exactly the numerator decay.
        values = affine_prefix_sum(
            torch.cat((v * normalized_key, normalized_key.abs()), -1),
            torch.cat((gate, gate), -1),
            initial_v,
        )
        numerator, denominator = values.chunk(2, -1)
        result = q * (numerator / denominator.clamp_min(self.eps)).to(q.dtype)
        if cache is not None and x.shape[1]:
            cache.write_state(self.layer_id, keys[:, -1:], values[:, -1:], x.shape[1])
        return result
