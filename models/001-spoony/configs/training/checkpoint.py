from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/wikipedia-en-600m/checkpoints"),
    interval=10000,
)
