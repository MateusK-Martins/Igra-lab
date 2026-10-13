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


class SoftRMSNormImpl(nn.Module):
    """Compress feature RMS smoothly; do not amplify small input vectors.

    No learned gain: the denominator is at least one and the Jacobian at zero
    is the identity. This changes only the internal cumulative normalizations.
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        working = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean_square = working.square().mean(dim=-1, keepdim=True)
        return (working * torch.rsqrt(1.0 + mean_square)).to(x.dtype)
