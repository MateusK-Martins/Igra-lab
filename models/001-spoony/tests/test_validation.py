import math
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config.types import PrecisionConfig
from src.training.precision import build_precision
from src.training.validation import validate


@pytest.mark.parametrize("training", [True, False])
def test_validation_weights_tokens_disables_dropout_and_preserves_weights_and_gradients(
    training,
):
    inputs = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    targets = (inputs + 1) % 3
    table = torch.tensor([[2.0, -1, 0], [0, 3, -2], [-1, 0, 1]])
    model = torch.nn.Sequential(torch.nn.Embedding(3, 3), torch.nn.Dropout(1.0))
    with torch.no_grad():
        model[0].weight.copy_(table)
    model.train(training)
    model[0].weight.grad = torch.ones_like(model[0].weight)
    before = model[0].weight.detach().clone()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    observed = []
    hook = model.register_forward_hook(
        lambda module, args, output: observed.append(
            (module.training, torch.is_grad_enabled(), output.requires_grad)
        )
    )
    try:
        loss, perplexity = validate(model, loader, nullcontext(), torch.device("cpu"))
    finally:
        hook.remove()
    log_probs = F.log_softmax(table[inputs], dim=-1)
    expected = -log_probs.gather(-1, targets.unsqueeze(-1)).mean().item()
    assert loss == pytest.approx(expected)
    assert perplexity == pytest.approx(math.exp(expected))
    assert observed == [(False, False, False)] * 3
    assert model.training is training
    torch.testing.assert_close(model[0].weight, before)
    torch.testing.assert_close(model[0].weight.grad, torch.ones_like(before))


@pytest.mark.parametrize("training", [True, False])
def test_empty_validation_rejects_and_restores_mode(training):
    model = torch.nn.Linear(2, 3)
    model.train(training)
    loader = DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    )
    with pytest.raises(ValueError, match="target tokens"):
        validate(model, loader, nullcontext(), torch.device("cpu"))
    assert model.training is training


@pytest.mark.parametrize("training", [True, False])
def test_validation_restores_mode_when_forward_fails(training):
    model = torch.nn.Embedding(2, 3)
    model.train(training)
    loader = DataLoader(TensorDataset(torch.tensor([[9]]), torch.tensor([[0]])))
    with pytest.raises(IndexError):
        validate(model, loader, nullcontext(), torch.device("cpu"))
    assert model.training is training


def test_perplexity_overflow_returns_infinity():
    model = torch.nn.Embedding(1, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0, 1000.0]]))
    loader = DataLoader(TensorDataset(torch.tensor([[0]]), torch.tensor([[0]])))
    loss, perplexity = validate(model, loader, nullcontext(), torch.device("cpu"))
    assert loss == pytest.approx(1000)
    assert perplexity == float("inf")


@pytest.mark.parametrize(
    "device,mode",
    [
        ("cpu", "fp32"),
        ("cpu", "bf16"),
        ("cuda", "fp32"),
        ("cuda", "bf16"),
        ("cuda", "fp16"),
    ],
)
def test_validation_runs_with_precision_context_and_moves_cpu_batches(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and mode == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    model = torch.nn.Sequential(torch.nn.Embedding(5, 4), torch.nn.Linear(4, 5)).to(
        device
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[0, 1], [2, 3]]), torch.tensor([[1, 2], [3, 4]]))
    )
    context, _ = build_precision(torch.device(device), PrecisionConfig(mode))
    first = validate(model, loader, context, torch.device(device))
    second = validate(model, loader, context, torch.device(device))
    assert all(math.isfinite(value) for value in first)
    assert first == pytest.approx(second)
    assert all(p.grad is None for p in model.parameters())
    assert model.training
