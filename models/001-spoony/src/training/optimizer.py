from torch import nn
from torch.optim import AdamW

from src.config.types import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
