from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/a4-rope-14m") / "checkpoints",
    interval=1000,
)
