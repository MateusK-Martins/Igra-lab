from src.config.paths import data_path
from src.config.types import DataLoaderConfig
from src.sft.config import SFTDataConfig

RAW_DIRECTORY = data_path("raw/comparison-30m/squad")
OUTPUT_DIRECTORY = data_path("processed/squad-full-v1")
REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
SELECTION_SEED = 42
VALIDATION_LIMIT = 1000

data_config = SFTDataConfig(
    train_path=OUTPUT_DIRECTORY / "train.jsonl",
    validation_path=OUTPUT_DIRECTORY / "validation.jsonl",
    tokenizer_path=data_path("processed/comparison-30m/tokenizer.json"),
    context_length=512,
    instruction_prefix="Context:\n",
    response_prefix="\nAnswer: ",
)
train_loader_config = DataLoaderConfig(8, True, 0, True, False, 42)
validation_loader_config = DataLoaderConfig(8, False, 0, True, False, 42)
