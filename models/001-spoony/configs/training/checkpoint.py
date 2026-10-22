from pathlib import Path

from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    results_path("finetuning/lantern70m-instruct-tinystories1m/checkpoints"),
    25,
    keep_last=1,
    backup_directory=Path(
        "/home/mateus/Documents/Igra-lab-checkpoints/lantern70m-instruct-tinystories1m"
    ),
)
