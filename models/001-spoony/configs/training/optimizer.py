from src.config.types import OptimizerConfig

optimizer_config = OptimizerConfig(
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.01,
)
