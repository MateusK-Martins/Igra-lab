"""Learning-rate schedule indexed by optimizer updates."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.config.types import SchedulerConfig


def build_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LambdaLR:
    """Set the first update's LR now; call step() after each optimizer.step().

    Warmup updates use factors 1/warmup_steps through 1. Cosine decay reaches
    min_lr_ratio on the final planned update and stays there afterwards.
    Without warmup, the first update uses the base LR (also for total_steps=1).
    Restore optimizer and scheduler states after rebuilding both with the same
    configuration; LambdaLR does not serialize this function's configuration.
    """

    def lr_factor(step: int) -> float:
        if step >= config.total_steps:
            return config.min_lr_ratio
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps

        if config.warmup_steps:
            progress = (step + 1 - config.warmup_steps) / (
                config.total_steps - config.warmup_steps
            )
        else:
            progress = step / max(1, config.total_steps - 1)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_factor)
