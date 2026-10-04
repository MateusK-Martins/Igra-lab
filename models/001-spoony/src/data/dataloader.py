from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.config.types import ChunkingConfig, DataLoaderConfig
from src.data.chunker import StoredTokenSequenceDataset
from src.data.serialization import TokenStore


def build_dataloader(
    dataset: StoredTokenSequenceDataset, config: DataLoaderConfig
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        generator=generator,
    )


def build_split_dataloader(
    split_directory: Path,
    chunking_config: ChunkingConfig,
    loader_config: DataLoaderConfig,
) -> DataLoader[tuple[Tensor, Tensor]]:
    store = TokenStore(split_directory)

    dataset = StoredTokenSequenceDataset(
        store=store,
        context_length=chunking_config.context_length,
        stride=chunking_config.stride,
    )

    return build_dataloader(dataset, loader_config)
