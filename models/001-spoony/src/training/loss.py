import torch.nn.functional as F
from torch import Tensor


def next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
