from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path(
        "pretraining/lantern70-tokenmlp-shared64-wikipedia100m-v1/checkpoints"
    ),
    interval=1000,
)
