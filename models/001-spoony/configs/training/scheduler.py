"""Editable schedule values; steps count optimizer updates, not batches."""

from src.config.types import SchedulerConfig

scheduler_config = SchedulerConfig(
    warmup_steps=100,
    total_steps=1000,
    min_lr_ratio=0.1,
)
