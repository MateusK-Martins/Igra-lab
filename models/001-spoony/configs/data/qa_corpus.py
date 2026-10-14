from src.config.paths import data_path
from src.sft.config import SFTDataConfig

qa_config = SFTDataConfig(
    train_path=data_path("processed/comparison-30m/qa-train.jsonl"),
    validation_path=data_path("processed/comparison-30m/qa-validation.jsonl"),
    tokenizer_path=data_path("processed/comparison-30m/tokenizer.json"),
    context_length=512,
    instruction_prefix="Context:\n",
    response_prefix="\nAnswer: ",
)

GENERATION_EXAMPLES = 256
MAX_NEW_TOKENS = 64
