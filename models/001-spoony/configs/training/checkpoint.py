from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/attentionless-sigmoid-rms-v2/checkpoints"),
    interval=1000,
)
