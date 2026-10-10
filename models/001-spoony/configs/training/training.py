from src.config.types import TrainingConfig

training_config = TrainingConfig(
    accumulation_steps=1,
    max_grad_norm=1.0,
    validation_interval=100,
)
