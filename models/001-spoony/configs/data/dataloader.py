from src.config.types import ChunkingConfig, DataLoaderConfig

train_loader_config = DataLoaderConfig(
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    seed=42,
)

validation_loader_config = DataLoaderConfig(
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    seed=42,
)

chunking_config = ChunkingConfig(
    context_length=128,
    stride=128,
)
