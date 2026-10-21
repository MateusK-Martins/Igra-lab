from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path(
        "pretraining/lantern-30m-tokenmlp-projection-only-v1/checkpoints"
    ),
    interval=2000,
)
