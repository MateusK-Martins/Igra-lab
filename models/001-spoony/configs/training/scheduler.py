"""Editable schedule values; steps count optimizer updates, not batches."""

from src.config.types import SchedulerConfig

scheduler_config = SchedulerConfig(
    warmup_steps=200,
    total_steps=6500,
    min_lr_ratio=0.1,
)
