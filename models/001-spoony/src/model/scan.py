"""Differentiable parallel scans for vector affine recurrences."""

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


def affine_prefix_sum(
    update: Tensor, decay: Tensor, initial: Tensor | None = None
) -> Tensor:
    """Parallel affine scan for token-dependent [B,T,D] diagonal coefficients.

    Compose (A_right,B_right) after (A_left,B_left) as
    (A_right*A_left, B_right + A_right*B_left). Every doubling stage uses
    the previous stage's tensors; time order matters for the additive term.
    """
    a = decay
    b = update + decay * 0 if update.shape[1] <= 1 else update
    offset = 1
    while offset < update.shape[1]:
        next_b = torch.cat(
            (b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), dim=1
        )
        a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
        b = next_b
        offset *= 2
    return b if initial is None else b + a * initial
