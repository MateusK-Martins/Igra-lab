"""Feed-forward modules operating on the last tensor dimension."""

from torch import Tensor, nn
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage


class SwiGLUImpl(nn.Module):
    """Apply down(silu(gate(x)) * up(x)); residuals are handled externally."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(input_features, hidden_features, bias=bias)
        self.up = nn.Linear(input_features, hidden_features, bias=bias)
        self.down = nn.Linear(hidden_features, output_features, bias=bias)

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LinearImpl(nn.Linear):
    """Linear body block accepting the shared execution arguments."""

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        return super().forward(x)
