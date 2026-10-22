"""Edit tokenizer training settings and the processed-data destination here."""

from src.config.paths import data_path
from src.config.types import PreparationConfig, TokenizerConfig

preparation_config = PreparationConfig(
    tokenizer=TokenizerConfig(
        target_vocab_size=32768,
        min_frequency=2,
        special_tokens=("<|unk|>", "<|eot|>"),
    ),
    output_directory=data_path("processed/lantern30m-wikipedia-vocab32k"),
    end_of_document_token="<|eot|>",
)
