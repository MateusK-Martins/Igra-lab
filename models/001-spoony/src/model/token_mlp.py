"""Projection-only bottleneck branch used for the TokenMLP ablation."""

from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


class TokenMLPImpl(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: Tensor,
        *,
        token_projections: tuple[nn.Module, nn.Module],
        cache: CacheStorage | None = None,
    ) -> Tensor:
        del cache
        down, up = token_projections
        return up(down(x))
