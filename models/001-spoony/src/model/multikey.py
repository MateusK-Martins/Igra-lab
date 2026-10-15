"""Full-width decaying key banks, combined before the original global norm."""

import math

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.cumulative import CumulativeMixerImpl
from src.model.scan import affine_prefix_sum, decayed_prefix_sum


class MultiKeyMixerImpl(CumulativeMixerImpl):
    def __init__(
        self,
        features: int,
        num_keys: int,
        tied_query: bool,
        eps: float,
        decay_init: float,
    ):
        # Keep original Q/V, gate and downstream initialization unchanged.
        super().__init__(
            features,
            eps,
            bias=False,
            value_decay=True,
            decay_mode="token",
            decay_init=decay_init,
        )
        self.features, self.num_keys, self.tied_query = features, num_keys, tied_query
        original_key = self.k_proj.weight.detach().clone()
        with torch.random.fork_rng(devices=[]):
            self.k_proj = nn.Linear(features, num_keys * features, bias=False)
            with torch.no_grad():
                self.k_proj.weight[:features].copy_(original_key)
            self.key_decay_logits = nn.Parameter(
                torch.full(
                    (num_keys * features,), math.log(decay_init / (1 - decay_init))
                )
            )
            self.key_output_weights = nn.Parameter(torch.empty(num_keys, features))
            nn.init.normal_(self.key_output_weights, std=1 / math.sqrt(num_keys))
            # Construct in both modes so common initialization stays identical.
            reader = nn.Linear(features, features, bias=False)
            self.key_query_proj = None if tied_query else reader

    def assign_id(self, layer_id: int):
        self.layer_id = f"MultiKeyMixerImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Multi-key cached execution requires no_grad")
        q = self.q_proj(x).sigmoid()
        k = self.k_proj(x).tanh()
        v = self.v_proj(x).sigmoid()
        query = q if self.key_query_proj is None else self.key_query_proj(x).sigmoid()
        dtype = torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        initial_key = initial_value = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.key_sum.shape
                    != (x.shape[0], 1, self.num_keys * self.features)
                    or previous.value_sum.shape != (x.shape[0], 1, self.features)
                    or previous.key_sum.dtype != dtype
                    or previous.value_sum.dtype != dtype
                ):
                    raise ValueError("Recurrent cache batch, features or dtype changed")
                initial_key, initial_value = previous.key_sum, previous.value_sum
        keys = decayed_prefix_sum(
            k.to(dtype), self.key_decay_logits.to(dtype).sigmoid(), initial_key
        )
        heads = keys.reshape(*keys.shape[:-1], self.num_keys, self.features)
        combined = (heads * self.key_output_weights.to(dtype)).sum(-2)
        # The key-reading query gates the bank sum BEFORE global RMSNorm.
        normalized = self.key_norm(query.to(dtype) * combined)
        values = affine_prefix_sum(
            v.to(dtype) * normalized, self.value_decay_gate(x).to(dtype), initial_value
        )
        result = q * self.value_norm(values).to(q.dtype)
        if cache is not None and x.shape[1]:
            cache.write_state(self.layer_id, keys[:, -1:], values[:, -1:], x.shape[1])
        return result
