from src.config.paths import results_path
from src.config.types import CheckpointConfig, LoggingConfig, TrainingConfig

training_config = TrainingConfig(1, 1.0, 20)
logging_config = LoggingConfig(results_path("sft/instructions-v1/tensorboard"), 10)
checkpoint_config = CheckpointConfig(
    results_path("sft/instructions-v1/checkpoints"), 20
)
