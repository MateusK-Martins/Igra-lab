from src.config.paths import results_path
from src.config.types import LoggingConfig, TrainingConfig

training_config = TrainingConfig(
    accumulation_steps=1,
    max_grad_norm=1.0,
    validation_interval=500,
)

logging_config = LoggingConfig(
    directory=results_path("pretraining/a4-14m/tensorboard"),
    interval=10,
)
