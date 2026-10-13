"""Differentiable parallel scans for diagonal, time-invariant recurrences."""

import torch
from torch import Tensor


def decayed_prefix_sum(
    x: Tensor, decay: Tensor | None = None, initial: Tensor | None = None
) -> Tensor:
    """Compute h[t] = decay * h[t-1] + x[t] for [B,T,D] inputs.

    Decay is [D], initial state is [B,1,D]. None means unit decay. The
    doubling scan has logarithmic sequential depth; every stage operates on
    all token positions in parallel. No inverse powers or division by decay
    are used, so near-zero coefficients remain numerically safe.
    """
    if decay is None:
        result = x.cumsum(dim=1)
        return result if initial is None else result + initial

    result = x + decay * 0 if x.shape[1] <= 1 else x
    power = decay
    offset = 1
    while offset < x.shape[1]:
        result = torch.cat(
            (result[:, :offset], result[:, offset:] + power * result[:, :-offset]),
            dim=1,
        )
        power = power * power
        offset *= 2
    if initial is not None:
        positions = torch.arange(1, x.shape[1] + 1, device=x.device)
        result = result + initial * decay.pow(positions[:, None])
    return result
