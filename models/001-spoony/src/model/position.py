"""Position functions"""

import math
from typing import Protocol

import torch
from torch import Tensor


class PositionRotation(Protocol):
    def __call__(self, x: Tensor, positions: Tensor) -> Tensor: ...


def rope(x: Tensor, positions: Tensor, *, base: float) -> Tensor:
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [B, H, T, D]")

    head_features = x.shape[-1]
    if head_features <= 0 or head_features % 2 != 0:
        raise ValueError("RoPE requires a positive, even head width")

    if positions.ndim != 1 or positions.shape[0] != x.shape[2]:
        raise ValueError("positions must have one entry per token")

    if type(base) not in (int, float) or not math.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and greater than zero")

    working = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

    feature_indices = torch.arange(
        0,
        head_features,
        2,
        device=x.device,
        dtype=working.dtype,
    )
    frequencies = base ** (-feature_indices / head_features)

    positions = positions.to(device=x.device, dtype=working.dtype)
    angles = positions[:, None] * frequencies[None, :]

    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]

    even = working[..., 0::2]
    odd = working[..., 1::2]

    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine

    rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)
