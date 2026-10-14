from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/comparison-fluxv-30m-v1/checkpoints"),
    interval=2000,
)
