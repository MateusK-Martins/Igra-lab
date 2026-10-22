"""Pinned English Wikipedia slice; no QA and no repeated documents."""

from src.config.paths import data_path

DATASET_ID = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SUBSET = "20231101.en"
RAW_DIRECTORY = data_path("raw/wikipedia-en")
TRAIN_TOKENS = 600_000_000
VALIDATION_TOKENS = 2_000_000
TOKENIZER_DOCUMENTS = 20_000
SPLIT_SEED = 42
SHARD_COUNT = 41

from src.config.paths import project_root

TOKENIZER_ARTIFACT = project_root() / "configs/tokenizers/wikipedia-en-32768.json"
EXPECTED_TOKENIZER_SHA256 = (
    "cc6dd6a41ed33ea342f3f1cc98a9fe7ea937fb75a32f00b908bdc12baf16f49b"
)
