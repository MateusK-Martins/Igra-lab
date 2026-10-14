"""Configured data loaders shared by benchmarking and the comparison runner."""

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.data.qa_corpus import qa_config
from configs.tokenizers.preparation import preparation_config
from configs.training.comparison import comparison_config
from src.data.dataloader import build_dataloader, build_split_dataloader
from src.data.mixture import BatchMixtureDataset
from src.data.tokenizer import ByteLevelBPETokenizer
from src.sft.data import SFTDataset


def loaders():
    directory = preparation_config.output_directory
    tokenizer = ByteLevelBPETokenizer.load(directory / "tokenizer.json")
    stories = build_split_dataloader(
        directory / "train", chunking_config, train_loader_config
    ).dataset
    qa = SFTDataset(qa_config.train_path, tokenizer, qa_config)
    mixed = BatchMixtureDataset(
        stories,
        qa,
        batch_size=train_loader_config.batch_size,
        story_batches_per_qa=comparison_config.story_batches_per_qa,
        seed=train_loader_config.seed,
    )
    train = build_dataloader(mixed, train_loader_config)
    story_val = build_split_dataloader(
        directory / "validation", chunking_config, validation_loader_config
    )
    qa_val = build_dataloader(
        SFTDataset(qa_config.validation_path, tokenizer, qa_config),
        validation_loader_config,
    )
    return tokenizer, train, story_val, qa_val
