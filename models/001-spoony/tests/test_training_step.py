import copy
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F

from src.config.types import SchedulerConfig
from src.training.scheduler import build_scheduler
from src.training.step import train_step


def test_accumulation_matches_combined_batch_loss_gradients_and_update():
    torch.manual_seed(12)
    model = torch.nn.Linear(4, 7).double()
    reference = copy.deepcopy(model)
    batches = [
        (torch.randn(n, 3, 4, dtype=torch.float64), torch.randint(7, (n, 3)))
        for n in (2, 1)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 5, 0.1))
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    inputs = torch.cat([x for x, _ in batches])
    targets = torch.cat([y for _, y in batches])
    expected_loss = F.cross_entropy(reference(inputs).transpose(1, 2), targets)
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.01)
    reference_optimizer.step()

    # An update must discard stale gradients from an earlier call.
    for param in model.parameters():
        param.grad = torch.full_like(param, 100)
    model.eval()
    loss, updated = train_step(
        model,
        batches,
        optimizer,
        scheduler,
        nullcontext(),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        0.01,
    )
    assert model.training and updated
    assert loss == pytest.approx(expected_loss.item(), abs=1e-6)
    assert scheduler.last_epoch == 1
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad)
        torch.testing.assert_close(actual, expected)
    assert (
        torch.linalg.vector_norm(
            torch.cat([p.grad.flatten() for p in model.parameters()])
        )
        <= 0.010001
    )


@pytest.mark.parametrize("max_norm", [0, -1, float("nan"), float("inf")])
def test_invalid_clipping_limit_rejected_without_updating(max_norm):
    model = torch.nn.Linear(4, 7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 3, 0))
    with pytest.raises(ValueError, match="max_grad_norm"):
        train_step(
            model,
            [(torch.ones(1, 1, 4), torch.zeros(1, 1, dtype=torch.long))],
            optimizer,
            scheduler,
            nullcontext(),
            torch.amp.GradScaler("cpu", enabled=False),
            torch.device("cpu"),
            max_norm,
        )
    assert scheduler.last_epoch == 0
    assert not optimizer.state


def test_empty_update_is_rejected():
    model = torch.nn.Linear(4, 7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 3, 0))
    with pytest.raises(ValueError, match="target tokens"):
        train_step(
            model,
            [],
            optimizer,
            scheduler,
            nullcontext(),
            torch.amp.GradScaler("cpu", enabled=False),
            torch.device("cpu"),
            1.0,
        )


def test_nonfinite_gradients_without_scaler_raise_before_optimizer_update():
    model = torch.nn.Linear(4, 7)
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 3, 0))
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            train_step(
                model,
                [(torch.ones(1, 1, 4), torch.zeros(1, 1, dtype=torch.long))],
                optimizer,
                scheduler,
                nullcontext(),
                torch.amp.GradScaler("cpu", enabled=False),
                torch.device("cpu"),
                1.0,
            )
    finally:
        hook.remove()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert scheduler.last_epoch == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fp16_overflow_skips_update_and_scheduler_then_recovers():
    from src.config.types import PrecisionConfig
    from src.training.precision import build_precision

    device = torch.device("cuda")
    model = torch.nn.Linear(4, 7).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 4, 0.1))
    context, scaler = build_precision(device, PrecisionConfig("fp16"))
    original = copy.deepcopy(model.state_dict())
    batches = [(torch.ones(1, 2, 4), torch.zeros(1, 2, dtype=torch.long))]
    scale = scaler.get_scale()
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        _, updated = train_step(
            model, batches, optimizer, scheduler, context, scaler, device, 1.0
        )
    finally:
        hook.remove()
    assert not updated
    assert scaler.get_scale() < scale
    assert scheduler.last_epoch == 0
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    _, updated = train_step(
        model, batches, optimizer, scheduler, context, scaler, device, 1.0
    )
    assert updated and scheduler.last_epoch == 1
    assert not torch.equal(model.weight, original["weight"])
