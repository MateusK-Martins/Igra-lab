from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path(
        "pretraining/lantern-tokenmlp-swiglu-gates-14m-v2/checkpoints"
    ),
    interval=1000,
)
