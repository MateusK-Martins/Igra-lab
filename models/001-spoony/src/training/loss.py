import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


def supervised_token_count(targets: Tensor) -> int:
    return int((targets != IGNORE_INDEX).sum().item())


def next_token_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
