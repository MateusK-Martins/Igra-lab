from src.config.paths import results_path
from src.config.types import CheckpointConfig, LoggingConfig, TrainingConfig

training_config = TrainingConfig(1, 1.0, 1000)
logging_config = LoggingConfig(results_path("sft/fluxv-30m-qa-v1/tensorboard"), 25)
checkpoint_config = CheckpointConfig(
    results_path("sft/fluxv-30m-qa-v1/checkpoints"), 2000
)
