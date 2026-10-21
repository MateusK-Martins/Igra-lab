from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/lantern-token-mlp-all-shared64-v1/checkpoints"),
    interval=1000,
)
