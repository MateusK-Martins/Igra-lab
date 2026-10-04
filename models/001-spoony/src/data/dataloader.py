import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.config.types import DataLoaderConfig
from src.data.chunker import StoredTokenSequenceDataset


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
