import math
from contextlib import AbstractContextManager

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.loss import next_token_loss, supervised_token_count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    context: AbstractContextManager,
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = 0

    try:
        for inputs, targets in loader:
            tokens = supervised_token_count(targets)
            if tokens == 0:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)

            with context:
                logits = model(inputs)
                loss = next_token_loss(logits, targets)

            total_loss += loss.double() * tokens
            total_tokens += tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Validation requires target tokens")

    mean_loss = (total_loss / total_tokens).item()

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    return mean_loss, perplexity
