"""Named, reviewable configurations for this model's experiments."""

from src.config.types import ChunkingConfig, DataConfig, TokenizerConfig


def tinystories_v2() -> DataConfig:
    """Return the initial data settings for the TinyStories validation run."""
    return DataConfig(
        tokenizer=TokenizerConfig(
            target_vocab_size=4096,
            min_frequency=2,
            special_tokens=("<|unk|>", "<|eot|>"),
        ),
        chunking=ChunkingConfig(context_length=128, stride=128),
    )
