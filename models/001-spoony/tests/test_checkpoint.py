import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.config.errors import ConfigError
from src.config.types import (
    CheckpointConfig,
    LoggingConfig,
    PrecisionConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.loop import train_loop
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 8)
        self.dropout = torch.nn.Dropout(0.3)
        self.head = torch.nn.Linear(8, 17, bias=False)
        self.head.weight = self.embedding.weight
        self.seen = []

    def forward(self, inputs):
        if self.training:
            self.seen.append(inputs.detach().cpu().clone())
        return self.head(self.dropout(self.embedding(inputs)))


def setup(mode="fp32", device="cpu"):
    torch.manual_seed(77)
    model = RecordingModel().to(device)
    inputs = torch.arange(44).reshape(11, 4) % 17
    targets = (inputs + 1) % 17
    train = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(123),
    )
    val = DataLoader(
        TensorDataset(inputs[:5], targets[:5]),
        batch_size=2,
        generator=torch.Generator().manual_seed(456),
    )
    scheduler_config = SchedulerConfig(2, 7, 0.1)
    training_config = TrainingConfig(2, 1.0, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_scheduler(optimizer, scheduler_config)
    context, scaler = build_precision(torch.device(device), PrecisionConfig(mode))
    if mode == "fp16":
        scaler = torch.amp.GradScaler("cuda", init_scale=16, growth_interval=2)
    args = {
        "model": model,
        "train_loader": train,
        "validation_loader": val,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "context": context,
        "scaler": scaler,
        "device": torch.device(device),
        "config": training_config,
        "total_steps": 7,
        "writer": MagicMock(spec=SummaryWriter),
        "logging_config": LoggingConfig(Path("unused-logs"), 1),
    }
    settings = {
        "scheduler": scheduler_config,
        "training": training_config,
        "precision": PrecisionConfig(mode),
        "dataset": "fixture-v1",
        "tokenizer": "fixture-v1",
    }
    return args, settings


def components(args, settings):
    return {
        key: args[key]
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "train_loader",
            "validation_loader",
        )
    } | {"settings": settings}


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


class Interrupted(Exception):
    pass


@pytest.mark.parametrize(
    "cut,mode,device",
    [
        (1, "fp32", "cpu"),
        (3, "fp32", "cpu"),
        (4, "fp32", "cpu"),
        (2, "fp16", "cuda"),
        (3, "bf16", "cuda"),
    ],
)
def test_resume_matches_continuous_shuffle_dropout_weights_rng_and_updates(
    tmp_path, cut, mode, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    baseline, _ = setup(mode, device)
    expected_state = train_loop(**baseline)
    expected_rng = torch.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state_all()
    interrupted, settings = setup(mode, device)
    checkpoints = CheckpointConfig(tmp_path, 1)

    def save_and_interrupt(path, **kwargs):
        save_checkpoint(path, **kwargs)
        if kwargs["state"].updates == cut:
            raise Interrupted

    with (
        patch("src.training.loop.save_checkpoint", side_effect=save_and_interrupt),
        pytest.raises(Interrupted),
    ):
        train_loop(
            **interrupted,
            checkpoint_config=checkpoints,
            checkpoint_settings=settings,
        )
    prefix = interrupted["model"].seen
    resumed, resume_settings = setup(mode, device)
    state = load_checkpoint(
        tmp_path / f"step-{cut:08d}.pt", **components(resumed, resume_settings)
    )
    result = train_loop(
        **resumed,
        state=state,
        checkpoint_config=checkpoints,
        checkpoint_settings=resume_settings,
    )
    assert result == expected_state
    assert_tree_equal(prefix + resumed["model"].seen, baseline["model"].seen)
    for key in ("model", "optimizer", "scheduler", "scaler"):
        assert_tree_equal(resumed[key].state_dict(), baseline[key].state_dict())
    assert resumed["model"].head.weight is resumed["model"].embedding.weight
    assert_tree_equal(
        resumed["train_loader"].generator.get_state(),
        baseline["train_loader"].generator.get_state(),
    )
    assert_tree_equal(
        resumed["validation_loader"].generator.get_state(),
        baseline["validation_loader"].generator.get_state(),
    )
    assert_tree_equal(torch.get_rng_state(), expected_rng)
    assert_tree_equal(torch.cuda.get_rng_state_all(), expected_cuda_rng)
    expected_logs = [
        c for c in baseline["writer"].add_scalar.call_args_list if c.args[2] > cut
    ]
    assert resumed["writer"].add_scalar.call_args_list == expected_logs
    payload = torch.load(
        tmp_path / "step-00000007.pt", weights_only=True, map_location="cpu"
    )
    assert payload["settings"]["scheduler"]["total_steps"] == 7


def test_failed_save_preserves_previous_checkpoint_and_removes_temporary_file(tmp_path):
    args, settings = setup()
    state = train_loop(**args)
    path = tmp_path / "checkpoint.pt"
    kwargs = components(args, settings) | {"state": state}
    save_checkpoint(path, **kwargs)
    previous = path.read_bytes()
    with (
        patch("src.training.checkpoint.torch.save", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        save_checkpoint(path, **kwargs)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", ["settings", "loader", "version", "cursor"])
def test_invalid_resume_is_rejected_before_model_mutation(tmp_path, change):
    args, settings = setup()
    state = train_loop(**args)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state=state, **components(args, settings))
    new_args, new_settings = setup()
    if change == "settings":
        new_settings["scheduler"] = replace(new_settings["scheduler"], total_steps=8)
    elif change == "loader":
        new_args["train_loader"] = DataLoader(
            new_args["train_loader"].dataset,
            batch_size=1,
            generator=torch.Generator().manual_seed(123),
        )
    else:
        payload = torch.load(path, weights_only=True)
        if change == "version":
            payload["format_version"] = 99
        else:
            payload["training_state"]["batches_in_epoch"] = 999
        torch.save(payload, path)
    before = copy.deepcopy(new_args["model"].state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(path, **components(new_args, new_settings))
    assert_tree_equal(new_args["model"].state_dict(), before)


@pytest.mark.parametrize("invalid", ["workers", "generator"])
def test_unsupported_loader_rejected_before_training(tmp_path, invalid):
    args, settings = setup()
    if invalid == "workers":
        args["train_loader"].num_workers = 1
    else:
        args["train_loader"].generator = None
    with pytest.raises(ValueError, match="num_workers|generator"):
        train_loop(
            **args,
            checkpoint_config=CheckpointConfig(tmp_path, 1),
            checkpoint_settings=settings,
        )
    assert args["model"].seen == []


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_checkpoint_config_rejects_invalid_interval(tmp_path, interval):
    with pytest.raises(ConfigError, match="interval"):
        CheckpointConfig(tmp_path, interval)


def test_checkpoint_config_rejects_nonpath_directory():
    with pytest.raises(ConfigError, match="directory"):
        CheckpointConfig("checkpoints", 1)


def test_periodic_final_save_skips_failed_updates_and_records_normalized_cursor(
    tmp_path,
):
    args, settings = setup()
    args["total_steps"] = 3
    observed = []

    def record(path, **kwargs):
        state = kwargs["state"]
        observed.append(
            (
                path.name,
                state.updates,
                state.batches,
                state.batches_in_epoch,
                state.completed_epochs,
            )
        )
        assert state.last_validation_loss == 1.0

    with (
        patch(
            "src.training.loop.train_step",
            side_effect=[(1.0, True), (1.0, False), (1.0, True), (1.0, True)],
        ),
        patch("src.training.loop.validate", return_value=(1.0, 2.0)),
        patch("src.training.loop.save_checkpoint", side_effect=record),
    ):
        train_loop(
            **args,
            checkpoint_config=CheckpointConfig(tmp_path, 2),
            checkpoint_settings=settings,
        )
    assert observed == [
        ("step-00000002.pt", 2, 6, 0, 1),
        ("step-00000003.pt", 3, 8, 2, 1),
    ]


def test_checkpoint_requires_explicit_settings(tmp_path):
    args, _ = setup()
    with pytest.raises(ValueError, match="settings"):
        train_loop(**args, checkpoint_config=CheckpointConfig(tmp_path, 1))
