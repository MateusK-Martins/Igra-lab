"""Reproducible qualitative probes of the final pretrained model."""

from configs.data.dataloader import chunking_config
from src.config.paths import results_path

BUNDLE = results_path("finetuning/lantern70m-instruct-tinystories1m/model")
OUTPUT = results_path("finetuning/lantern70m-instruct-tinystories1m/samples.json")
DEVICE = "cuda"
SEED = 42
MAX_NEW_TOKENS = 128
CONTEXT_LENGTH = chunking_config.context_length
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = (
    "Once upon a time, there was a little rabbit who",
    "Lily found a shiny key in the garden. She",
    "One day, a small robot lost its way. It",
    "Tom wanted to help his friend, but",
    "The little bird was afraid to fly. Then",
    "User: Tell me a short story about a kind dragon.\nAssistant: ",
)
