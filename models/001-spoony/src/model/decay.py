"""Token-conditioned feature-wise retention gates."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TokenDecayImpl(nn.Module):
    def __init__(self, features: int, initial_decay: float) -> None:
        super().__init__()
        # Constant initial retention matches the diagonal baseline. Zero-weight
        # initialization adds no RNG draws to the surrounding model recipe.
        self.weight = nn.Parameter(torch.zeros(features, features))
        self.bias = nn.Parameter(
            torch.full((features,), math.log(initial_decay / (1 - initial_decay)))
        )

    def forward(self, x: Tensor) -> Tensor:
        projected = F.linear(x, self.weight)
        dtype = (
            torch.float32
            if projected.dtype in (torch.float16, torch.bfloat16)
            else projected.dtype
        )
        # Add the bias and sigmoid in accumulation precision, including under
        # autocast, so the initial 0.99 coefficient is not rounded to BF16.
        return (projected.to(dtype) + self.bias.to(dtype)).sigmoid()
