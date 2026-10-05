import copy
import math

import pytest
import torch
from torch.optim import AdamW

from src.config.errors import ConfigError
from src.config.types import SchedulerConfig
from src.training.scheduler import build_scheduler


def make_optimizer():
    return AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)


def used_rates(optimizer, scheduler, count):
    rates = []
    for _ in range(count):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


def test_warmup_cosine_rates_used_by_each_optimizer_update():
    optimizer = make_optimizer()
    scheduler = build_scheduler(optimizer, SchedulerConfig(2, 6, 0.1))
    rates = used_rates(optimizer, scheduler, 8)
    root_half = math.sqrt(0.5)
    expected = [
        0.5,
        1,
        0.55 + 0.45 * root_half,
        0.55,
        0.55 - 0.45 * root_half,
        0.1,
        0.1,
        0.1,
    ]
    assert rates == pytest.approx([0.01 * factor for factor in expected])


@pytest.mark.parametrize(
    "config,expected",
    [
        (SchedulerConfig(0, 3, 0), [1, 0.5, 0, 0]),
        (SchedulerConfig(0, 1, 0.1), [1, 0.1, 0.1, 0.1]),
        (SchedulerConfig(1, 2, 0.1), [1, 0.1, 0.1, 0.1]),
        (SchedulerConfig(0, 3, 1), [1, 1, 1, 1]),
        (SchedulerConfig(2, 3, 1), [0.5, 1, 1, 1]),
    ],
)
def test_schedule_boundaries(config, expected):
    optimizer = make_optimizer()
    scheduler = build_scheduler(optimizer, config)
    assert used_rates(optimizer, scheduler, 4) == pytest.approx(
        [0.01 * factor for factor in expected]
    )


def test_multiple_parameter_groups_preserve_base_lr_ratio():
    optimizer = AdamW(
        [
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.01},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.02},
        ]
    )
    scheduler = build_scheduler(optimizer, SchedulerConfig(2, 6, 0.1))
    for _ in range(8):
        assert optimizer.param_groups[1]["lr"] == pytest.approx(
            2 * optimizer.param_groups[0]["lr"]
        )
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("completed_updates", [1, 3, 6])
def test_restored_scheduler_continues_same_rates_and_parameter_updates(
    completed_updates,
):
    config = SchedulerConfig(2, 6, 0.1)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = AdamW([parameter], lr=0.01)
    scheduler = build_scheduler(optimizer, config)
    for _ in range(completed_updates):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = AdamW([restored_parameter], lr=0.01)
    restored_scheduler = build_scheduler(restored_optimizer, config)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for _ in range(4):
        assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
        for opt, sched, param in (
            (optimizer, scheduler, parameter),
            (restored_optimizer, restored_scheduler, restored_parameter),
        ):
            param.grad = torch.ones_like(param)
            opt.step()
            sched.step()
        torch.testing.assert_close(restored_parameter, parameter)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_steps", 0),
        ("total_steps", -1),
        ("total_steps", True),
        ("total_steps", 1.5),
        ("warmup_steps", -1),
        ("warmup_steps", 6),
        ("warmup_steps", 7),
        ("warmup_steps", True),
        ("warmup_steps", 0.5),
        ("min_lr_ratio", -0.1),
        ("min_lr_ratio", 1.1),
        ("min_lr_ratio", float("nan")),
        ("min_lr_ratio", float("inf")),
        ("min_lr_ratio", True),
        ("min_lr_ratio", "0.1"),
    ],
)
def test_scheduler_config_rejects_invalid_settings(field, value):
    settings = {"warmup_steps": 2, "total_steps": 6, "min_lr_ratio": 0.1}
    settings[field] = value
    with pytest.raises(ConfigError, match=field):
        SchedulerConfig(**settings)


def test_editable_scheduler_config_builds():
    from configs.training.scheduler import scheduler_config

    optimizer = make_optimizer()
    scheduler = build_scheduler(optimizer, scheduler_config)
    assert scheduler.get_last_lr() == pytest.approx(
        [0.01 / scheduler_config.warmup_steps]
    )
