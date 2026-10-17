from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/a16-parallel-2head-3layer-14m") / "checkpoints",
    interval=1000,
)
