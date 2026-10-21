from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/lantern-30m-token-mlp-shared64-v1/checkpoints"),
    interval=2000,
)
