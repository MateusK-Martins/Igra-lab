import math
from contextlib import AbstractContextManager

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.training.loss import next_token_loss


def train_step(
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: LambdaLR,
    context: AbstractContextManager,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float,
) -> tuple[float, bool]:
    total_tokens = sum(targets.numel() for _, targets in batches)

    if total_tokens == 0:
        raise ValueError("Training step requires target tokens")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    mean_loss = torch.zeros((), device=device)

    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with context:
            logits = model(inputs)
            loss = next_token_loss(logits, targets)

        weighted_loss = loss * (targets.numel() / total_tokens)
        scaler.scale(weighted_loss).backward()
        mean_loss += weighted_loss.detach()

    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
        error_if_nonfinite=not scaler.is_enabled(),
    )

    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()

    did_update = scaler.get_scale() >= previous_scale
    if did_update:
        scheduler.step()

    return mean_loss.item(), did_update
