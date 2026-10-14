from contextlib import AbstractContextManager
from itertools import islice

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config.types import CheckpointConfig, LoggingConfig, TrainingConfig
from src.training.checkpoint import loader_signature, save_checkpoint, training_iterator
from src.training.loss import supervised_token_count
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
    writer: SummaryWriter,
    logging_config: LoggingConfig,
    state: TrainingState | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    checkpoint_settings: dict[str, object] | None = None,
) -> TrainingState:
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader must contain batches")

    if checkpoint_config is not None:
        loader_signature(train_loader)
        loader_signature(validation_loader)
        if checkpoint_settings is None:
            raise ValueError(
                "Checkpoint settings must be supplied when saving checkpoints"
            )

    state = TrainingState() if state is None else state

    while state.updates < total_steps:
        iterator = training_iterator(train_loader, state)

        while state.updates < total_steps:
            batches: list[tuple[Tensor, Tensor]] = list(
                islice(iterator, config.accumulation_steps)
            )

            if not batches:
                break

            learning_rate = optimizer.param_groups[0]["lr"]

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
            state.supervised_tokens += sum(
                supervised_token_count(targets) for _, targets in batches
            )
            state.updates += int(did_update)
            state.batches_in_epoch += len(batches)
            epoch_finished = state.batches_in_epoch == len(train_loader)
            if epoch_finished:
                # Exhaust the sampler as uninterrupted DataLoader traversal would.
                next(iterator, None)
                state.completed_epochs += 1
                state.batches_in_epoch = 0
                state.epoch_generator_state = None

            if did_update and (
                state.updates % logging_config.interval == 0
                or state.updates == total_steps
            ):
                writer.add_scalar("train/loss", loss, state.updates)
                writer.add_scalar("train/learning_rate", learning_rate, state.updates)
                writer.add_scalar("train/tokens", state.tokens, state.updates)

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

                writer.add_scalar("validation/loss", validation_loss, state.updates)
                writer.add_scalar("validation/perplexity", perplexity, state.updates)

            if (
                checkpoint_config is not None
                and did_update
                and (
                    state.updates % checkpoint_config.interval == 0
                    or state.updates == total_steps
                )
            ):
                save_checkpoint(
                    checkpoint_config.directory / f"step-{state.updates:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    state=state,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    settings=checkpoint_settings,
                )

            if epoch_finished:
                break

    return state
