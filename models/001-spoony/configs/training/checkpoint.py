from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/t16/checkpoints"),
    interval=1000,
)
