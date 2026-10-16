"""Causal fixed-window gMLP. Residual addition belongs to the outer wrapper."""

import torch
from torch import nn
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage


class GMLPImpl(nn.Module):
    def __init__(self, features, hidden_features, window_size):
        super().__init__()
        self.window_size = window_size
        self.channels = hidden_features // 2
        self.up = nn.Linear(features, hidden_features)
        self.norm = nn.LayerNorm(self.channels)
        self.spatial_weight = nn.Parameter(torch.empty(window_size, window_size))
        self.spatial_bias = nn.Parameter(torch.ones(window_size))
        nn.init.normal_(self.spatial_weight, std=1e-3 / window_size)
        self.down = nn.Linear(self.channels, features)

    def assign_id(self, layer_id):
        self.layer_id = f"GMLPImpl-w{self.window_size}-{layer_id}"

    def forward(self, x, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("External gMLP cache is inference-only")
        u, v = F.gelu(self.up(x)).chunk(2, dim=-1)
        v = self.norm(v)
        batch, length, channels = v.shape
        if length == 0:
            return self.down(u * v)
        offset = 0
        previous = None
        if cache is not None:
            state = cache.read_state(self.layer_id)
            offset = state.count % self.window_size
            if state.count:
                previous = state.value_sum.reshape(batch, self.window_size, channels)
        prefix = v.new_zeros(batch, offset, channels)
        if previous is not None:
            prefix = previous[:, :offset]
        joined = torch.cat((prefix, v), dim=1)
        padding = (-joined.shape[1]) % self.window_size
        windows = F.pad(joined, (0, 0, 0, padding)).reshape(
            batch, -1, self.window_size, channels
        )
        mixed = torch.einsum("ij,bnjc->bnic", self.spatial_weight.tril(), windows)
        mixed = mixed + self.spatial_bias[None, None, :, None]
        mixed = mixed.reshape(batch, -1, channels)[:, offset : offset + length]
        if cache is not None:
            # Fixed-size storage; only the current window's populated prefix is read.
            saved = windows[:, -1].reshape(batch, 1, -1)
            cache.write_state(self.layer_id, saved[:, :, :0], saved, length)
        return self.down(u * mixed)
