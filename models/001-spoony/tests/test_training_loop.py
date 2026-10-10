from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.errors import ConfigError
from src.config.types import SchedulerConfig, TrainingConfig
from src.training.loop import train_loop
from src.training.scheduler import build_scheduler
from src.training.state import TrainingState
from src.training.step import train_step
from src.training.validation import validate


def setup_loop(samples=5, batch_size=1):
    inputs = torch.arange(samples * 3).reshape(samples, 3) % 13
    targets = (inputs + 1) % 13
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    model = torch.nn.Sequential(torch.nn.Embedding(13, 4), torch.nn.Linear(4, 13))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = build_scheduler(optimizer, SchedulerConfig(0, 10, 0.1))
    return {
        "model": model,
        "train_loader": loader,
        "validation_loader": DataLoader(
            TensorDataset(inputs.flip(0), targets.flip(0)), batch_size=2
        ),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "context": nullcontext(),
        "scaler": torch.amp.GradScaler("cpu", enabled=False),
        "device": torch.device("cpu"),
    }


def test_real_loop_handles_partial_group_epoch_restart_and_short_last_batch():
    args = setup_loop(batch_size=2)
    with patch("src.training.loop.train_step", wraps=train_step) as step:
        state = train_loop(**args, config=TrainingConfig(2, 1.0, 100), total_steps=3)
    assert [len(call.kwargs["batches"]) for call in step.call_args_list] == [2, 1, 2]
    assert state.updates == 3
    assert state.batches == 5
    assert state.tokens == 27  # Nine examples, each with three target tokens.
    assert state.completed_epochs == 1
    assert state.last_loss is not None and torch.isfinite(torch.tensor(state.last_loss))
    assert args["scheduler"].last_epoch == 3
    expected = validate(
        args["model"], args["validation_loader"], args["context"], args["device"]
    )
    assert (
        state.last_validation_loss,
        state.last_validation_perplexity,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "accumulation,total,expected_batches,expected_epochs",
    [
        (1, 2, 2, 0),
        (1, 5, 5, 1),
        (2, 3, 5, 1),
        (2, 4, 7, 1),
        (8, 2, 10, 2),
    ],
)
def test_loop_stops_at_successful_update_limit(
    accumulation, total, expected_batches, expected_epochs
):
    args = setup_loop()
    with (
        patch("src.training.loop.train_step", return_value=(2.0, True)),
        patch("src.training.loop.validate", return_value=(1.5, 4.5)),
    ):
        state = train_loop(
            **args, config=TrainingConfig(accumulation, 1.0, 100), total_steps=total
        )
    assert state == TrainingState(
        total, expected_batches, expected_batches * 3, expected_epochs, 2.0, 1.5, 4.5
    )


def test_skipped_update_counts_consumed_data_but_not_successful_updates():
    args = setup_loop()
    with (
        patch(
            "src.training.loop.train_step",
            side_effect=[(3.0, False), (2.0, True), (1.0, True), (0.5, True)],
        ) as step,
        patch("src.training.loop.validate", return_value=(1.5, 4.5)),
    ):
        state = train_loop(**args, config=TrainingConfig(2, 1.0, 100), total_steps=3)
    assert step.call_count == 4
    assert state == TrainingState(3, 7, 21, 1, 0.5, 1.5, 4.5)


@pytest.mark.parametrize("total_steps", [0, -1, True, 1.5])
def test_loop_rejects_invalid_update_limit(total_steps):
    with pytest.raises(ValueError, match="total_steps"):
        train_loop(
            **setup_loop(), config=TrainingConfig(1, 1.0, 100), total_steps=total_steps
        )


def test_loop_rejects_empty_loader():
    with pytest.raises(ValueError, match="DataLoader"):
        train_loop(
            **setup_loop(samples=0), config=TrainingConfig(1, 1.0, 100), total_steps=1
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("accumulation_steps", 0),
        ("accumulation_steps", -1),
        ("accumulation_steps", True),
        ("accumulation_steps", 1.5),
        ("max_grad_norm", 0),
        ("max_grad_norm", -1),
        ("max_grad_norm", True),
        ("max_grad_norm", "1"),
        ("max_grad_norm", float("nan")),
        ("max_grad_norm", float("inf")),
    ],
)
def test_training_config_rejects_invalid_settings(field, value):
    settings = {
        "accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "validation_interval": 100,
    }
    settings[field] = value
    with pytest.raises(ConfigError, match=field):
        TrainingConfig(**settings)


@pytest.mark.parametrize(
    "total,interval,expected",
    [
        (5, 2, [2, 4, 5]),
        (4, 2, [2, 4]),
        (3, 100, [3]),
        (1, 100, [1]),
        (3, 1, [1, 2, 3]),
    ],
)
def test_validation_runs_at_completed_update_intervals_and_final_once(
    total, interval, expected
):
    args = setup_loop()
    observed = []

    def observe(**kwargs):
        observed.append(args["scheduler"].last_epoch)
        assert kwargs["loader"] is args["validation_loader"]
        return float(len(observed)), 10.0

    with patch("src.training.loop.validate", side_effect=observe):
        state = train_loop(
            **args, config=TrainingConfig(2, 1.0, interval), total_steps=total
        )
    assert observed == expected
    assert state.last_validation_loss == len(expected)
    assert state.last_validation_perplexity == 10.0


def test_validation_does_not_repeat_on_skips_at_interval_boundary():
    args = setup_loop()
    successful = 0
    observed = []
    outcomes = iter([True, True, False, False, True, True])

    def fake_step(**kwargs):
        nonlocal successful
        updated = next(outcomes)
        successful += int(updated)
        return 1.0, updated

    def observe(**kwargs):
        observed.append(successful)
        return 1.0, 2.0

    with (
        patch("src.training.loop.train_step", side_effect=fake_step),
        patch("src.training.loop.validate", side_effect=observe),
    ):
        state = train_loop(**args, config=TrainingConfig(1, 1.0, 2), total_steps=4)
    assert observed == [2, 4]
    assert state.updates == 4 and state.batches == 6


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, None, "100"])
def test_training_config_rejects_invalid_validation_interval(interval):
    with pytest.raises(ConfigError, match="validation_interval"):
        TrainingConfig(1, 1.0, interval)
