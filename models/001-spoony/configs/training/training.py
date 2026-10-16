from src.config.paths import results_path
from src.config.types import LoggingConfig, TrainingConfig

training_config = TrainingConfig(
    accumulation_steps=1,
    max_grad_norm=1.0,
    validation_interval=1000,
)

logging_config = LoggingConfig(
    directory=results_path("pretraining/tide-a16-30m-v1/tensorboard"),
    interval=25,
)
