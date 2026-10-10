from contextlib import AbstractContextManager
from itertools import islice

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config.types import TrainingConfig
from src.training.state import TrainingState
from src.training.step import train_step
from src.training.validation import validate


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    context: AbstractContextManager,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
    total_steps: int,
) -> TrainingState:
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader must contain batches")

    state = TrainingState()

    while state.updates < total_steps:
        iterator = iter(train_loader)

        while state.updates < total_steps:
            batches: list[tuple[Tensor, Tensor]] = list(
                islice(iterator, config.accumulation_steps)
            )

            if not batches:
                break

            loss, did_update = train_step(
                model=model,
                batches=batches,
                optimizer=optimizer,
                scheduler=scheduler,
                context=context,
                scaler=scaler,
                device=device,
                max_grad_norm=config.max_grad_norm,
            )

            state.last_loss = loss
            state.batches += len(batches)
            state.tokens += sum(targets.numel() for _, targets in batches)
            state.updates += int(did_update)

            if did_update and (
                state.updates % config.validation_interval == 0
                or state.updates == total_steps
            ):
                validation_loss, perplexity = validate(
                    model=model,
                    loader=validation_loader,
                    context=context,
                    device=device,
                )

                state.last_validation_loss = validation_loss
                state.last_validation_perplexity = perplexity

            if len(batches) < config.accumulation_steps:
                break

        if state.batches and state.batches % len(train_loader) == 0:
            state.completed_epochs += 1

    return state
