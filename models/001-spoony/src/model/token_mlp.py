"""Projection-only TokenMLP ablation with no token embeddings or matrix."""

from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


class TokenMLPImpl(nn.Module):
    """Encode to the bottleneck and decode directly, without a matrix step."""

    def forward(
        self,
        x: Tensor,
        *,
        token_projections: tuple[nn.Module, nn.Module],
        cache: CacheStorage | None = None,
    ) -> Tensor:
        down, up = token_projections
        return up(down(x))
