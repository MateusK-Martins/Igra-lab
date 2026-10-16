"""Differentiable parallel scans for vector affine recurrences."""

import torch
from torch import Tensor


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
