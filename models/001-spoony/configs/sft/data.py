from src.config.paths import data_path
from src.config.types import DataLoaderConfig
from src.sft.config import SFTDataConfig

# Supply independent JSONL files before running SFT. Keep the pretrained tokenizer.
data_config = SFTDataConfig(
    train_path=data_path("raw/sft/train.jsonl"),
    validation_path=data_path("raw/sft/validation.jsonl"),
    tokenizer_path=data_path("processed/tinystories/tokenizer.json"),
    context_length=256,
)
train_loader_config = DataLoaderConfig(8, True, 0, True, False, 42)
validation_loader_config = DataLoaderConfig(8, False, 0, True, False, 42)
