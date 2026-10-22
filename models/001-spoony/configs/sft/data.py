from src.config.paths import data_path
from src.config.types import DataLoaderConfig
from src.sft.config import SFTDataConfig

DIRECTORY = data_path("raw/sft/simple-instruct-1m")
data_config = SFTDataConfig(
    DIRECTORY / "train.jsonl",
    DIRECTORY / "validation.jsonl",
    DIRECTORY / "tokenizer.json",
    2048,
)
train_loader_config = DataLoaderConfig(2, True, 0, True, False, 42)
validation_loader_config = DataLoaderConfig(2, False, 0, True, False, 42)
