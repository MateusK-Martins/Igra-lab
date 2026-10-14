from src.config.paths import data_path
from src.sft.config import SFTDataConfig

DATASET_ID = "rajpurkar/squad"
REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
RAW_DIRECTORY = data_path("raw/comparison-30m/squad")
SELECTION_SEED = 42
QA_LIMITS = {"train": 30000, "validation": 1000}
SOURCE_FILES = {
    split: f"plain_text/{split}-00000-of-00001.parquet" for split in QA_LIMITS
}

qa_config = SFTDataConfig(
    train_path=data_path("processed/comparison-30m/qa-train.jsonl"),
    validation_path=data_path("processed/comparison-30m/qa-validation.jsonl"),
    tokenizer_path=data_path("processed/comparison-30m/tokenizer.json"),
    context_length=512,
    instruction_prefix="Context:\n",
    response_prefix="\nAnswer: ",
)
