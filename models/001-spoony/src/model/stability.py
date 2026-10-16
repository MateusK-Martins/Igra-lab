"""Stability modules operating on the last tensor dimension."""

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


class RMSNormImpl(nn.Module):
    def __init__(self, features: int, eps: float) -> None:
        super().__init__()
        self.features = features
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(features))

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:

        working = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        mean_square = working.square().mean(dim=-1, keepdim=True)
        normalized = working * torch.rsqrt(mean_square + self.eps)

        return (normalized * self.weights).to(dtype=x.dtype)
