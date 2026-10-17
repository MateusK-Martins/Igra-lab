from configs.model.model import RUN_NAME
from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path(f"pretraining/{RUN_NAME}/checkpoints"),
    interval=1000,
)
