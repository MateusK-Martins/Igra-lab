"""One pass through all complete Wikipedia windows; optimizer updates."""

from math import ceil

from configs.data.dataloader import chunking_config, train_loader_config
from configs.data.wikipedia import TRAIN_TOKENS
from configs.training.training import training_config
from src.config.types import SchedulerConfig

windows = (
    1 + (TRAIN_TOKENS - chunking_config.context_length - 1) // chunking_config.stride
)
batches = ceil(windows / train_loader_config.batch_size)
scheduler_config = SchedulerConfig(
    warmup_steps=1000,
    total_steps=ceil(batches / training_config.accumulation_steps),
    min_lr_ratio=0.1,
)
