"""Feed-forward modules operating on the last tensor dimension."""

from torch import Tensor, nn
from torch.nn import functional as F


class SwiGLUFeedForward(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
