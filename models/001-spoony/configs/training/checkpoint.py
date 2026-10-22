from configs.training.health import checkpoint_backup_directory
from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/wikipedia-en-600m/checkpoints"),
    interval=1000,
    keep_last=2,
    backup_directory=checkpoint_backup_directory,
)
