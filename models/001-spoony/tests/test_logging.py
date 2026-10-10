from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config.errors import ConfigError
from src.config.types import LoggingConfig
from src.training.logging import build_writer


def test_writer_persists_scalar_values_and_update_numbers(tmp_path):
    directory = tmp_path / "run" / "tensorboard"
    writer = build_writer(LoggingConfig(directory, 10))
    try:
        writer.add_scalar("train/loss", 2.5, 10)
        writer.add_scalar("train/loss", 1.5, 20)
        writer.add_scalar("validation/perplexity", 7.0, 20)
    finally:
        writer.close()
    events = EventAccumulator(str(directory)).Reload()
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [
        (10, 2.5),
        (20, 1.5),
    ]
    assert [(e.step, e.value) for e in events.Scalars("validation/perplexity")] == [
        (20, 7.0)
    ]


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, None, "10"])
def test_logging_config_rejects_invalid_interval(tmp_path, interval):
    with pytest.raises(ConfigError, match="interval"):
        LoggingConfig(tmp_path, interval)


@pytest.mark.parametrize("directory", ["logs", None, 3])
def test_logging_config_rejects_nonpath_directory(directory):
    with pytest.raises(ConfigError, match="directory"):
        LoggingConfig(directory, 10)


def test_editable_logging_config_is_valid():
    from configs.training.training import logging_config

    assert isinstance(logging_config.directory, Path)
    assert logging_config.interval > 0


def test_resumed_writer_discards_abandoned_future_events(tmp_path):
    config = LoggingConfig(tmp_path, 1)
    writer = build_writer(config)
    writer.add_scalar("train/loss", 2.0, 10)
    writer.add_scalar("train/loss", 9.0, 30)
    writer.close()
    writer = build_writer(config, resume_step=10)
    writer.add_scalar("train/loss", 1.0, 20)
    writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [
        (10, 2.0),
        (20, 1.0),
    ]
