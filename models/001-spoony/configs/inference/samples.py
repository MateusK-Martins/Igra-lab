"""Reproducible qualitative probes of the final pretrained model."""

from configs.data.dataloader import chunking_config
from src.config.paths import results_path

BUNDLE = results_path("sft/lantern70m-instruct1m-think3-consume/model")
OUTPUT = results_path("sft/lantern70m-instruct1m-think3-consume/samples.json")
DEVICE = "cuda"
SEED = 42
MAX_NEW_TOKENS = 128
CONTEXT_LENGTH = chunking_config.context_length
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = tuple(
    "User: " + instruction + "\nAssistant: "
    for instruction in (
        "What is the capital of France?",
        "What is 7 plus 5?",
        "Explain photosynthesis in one sentence.",
        "Write a short story about a lost robot.",
        "Pretend you are a pirate. Introduce yourself.",
        "List three fruits.",
    )
)
