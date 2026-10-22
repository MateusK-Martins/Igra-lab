"""Reproducible qualitative probes of the final pretrained model."""

from configs.data.dataloader import chunking_config
from src.config.paths import results_path

BUNDLE = results_path("pretraining/wikipedia-en-600m/model")
OUTPUT = results_path("pretraining/wikipedia-en-600m/samples.json")
DEVICE = "cuda"
SEED = 42
MAX_NEW_TOKENS = 128
CONTEXT_LENGTH = chunking_config.context_length
TEMPERATURE = 0.8
TOP_K = 40
PROMPTS = (
    "The Solar System consists of",
    "Photosynthesis is the process by which",
    "The Roman Empire was",
    "In mathematics, a prime number is",
    "The Amazon River is",
    "A computer processor is",
)
