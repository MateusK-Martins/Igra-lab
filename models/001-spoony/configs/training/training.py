from configs.model.model import RUN_NAME
from src.config.paths import results_path
from src.config.types import LoggingConfig, TrainingConfig

training_config = TrainingConfig(
    accumulation_steps=1,
    max_grad_norm=1.0,
    validation_interval=500,
)

logging_config = LoggingConfig(
    directory=results_path(f"pretraining/{RUN_NAME}/tensorboard"),
    interval=10,
)
