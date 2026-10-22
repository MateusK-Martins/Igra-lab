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

TOKENIZER_ARTIFACT = project_root() / "configs/tokenizers/wikipedia-en-8192.json"
EXPECTED_TOKENIZER_SHA256 = (
    "1d40abc4915bc2a585c3a52595a2565275a2cae2015476ff51d6aef40f617729"
)
