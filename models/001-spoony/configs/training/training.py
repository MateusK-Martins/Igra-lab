from src.config.paths import results_path
from src.config.types import LoggingConfig, TrainingConfig

training_config = TrainingConfig(4, 1.0, 25)
logging_config = LoggingConfig(
    results_path("finetuning/lantern70m-instruct-tinystories1m/tensorboard"), 5
)
