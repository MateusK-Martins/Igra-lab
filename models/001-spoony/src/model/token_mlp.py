"""Token-specific nonlinear matrices with tables and projections shared by layers."""

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.model.cache_storage import CacheStorage


def _fused_transform(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    # CUDA compilation fuses the outer product, tanh, multiply and reduction.
    # This evaluates the dense matrix exactly without writing it to device RAM.
    return (torch.tanh(a.unsqueeze(-1) * b.unsqueeze(-2)) * x.unsqueeze(-2)).sum(-1)


_cuda_transform = torch.compile(_fused_transform, fullgraph=True, dynamic=True)


class TokenMLPImpl(nn.Module):
    def __init__(self, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size

    @staticmethod
    def transform(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
        if x.is_cuda:
            return _cuda_transform(x, a, b)
        # Matrix rows are output features; columns are input features.
        matrix = torch.tanh(a.unsqueeze(-1) * b.unsqueeze(-2))
        return torch.matmul(matrix, x.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        x: Tensor,
        *,
        token_vectors: tuple[Tensor, Tensor],
        token_projections: tuple[nn.Module, nn.Module],
        cache: CacheStorage | None = None,
    ) -> Tensor:
        down, up = token_projections
        x = down(x)
        a, b = token_vectors
        if a.shape != x.shape or b.shape != x.shape:
            raise ValueError(
                "Token vectors must match the projected hidden-state shape"
            )
        flat_x = x.reshape(-1, x.shape[-1])
        flat_a = a.reshape_as(flat_x)
        flat_b = b.reshape_as(flat_x)
        if flat_x.shape[0] == 0:
            return up(x.clone())
        outputs = []
        for start in range(0, flat_x.shape[0], self.chunk_size):
            end = start + self.chunk_size
            args = (flat_x[start:end], flat_a[start:end], flat_b[start:end])
            if torch.is_grad_enabled() and any(t.requires_grad for t in args):
                output = checkpoint(self.transform, *args, use_reentrant=False)
            else:
                output = self.transform(*args)
            outputs.append(output)
        return up(torch.cat(outputs, dim=0).reshape_as(x))
