from pathlib import Path

from src.config.paths import results_path
from src.config.types import CheckpointConfig, LoggingConfig, TrainingConfig

training_config = TrainingConfig(4, 1.0, 250)
logging_config = LoggingConfig(
    results_path("sft/lantern70m-instruct1m-normal/tensorboard"), 25
)
checkpoint_config = CheckpointConfig(
    results_path("sft/lantern70m-instruct1m-normal/checkpoints"),
    250,
    keep_last=2,
    backup_directory=Path(
        "/home/mateus/Documents/Igra-lab-checkpoints/lantern70m-sft1m-normal"
    ),
)
