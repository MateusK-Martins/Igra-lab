"""Frozen TinyStories subset recipe; preparation time is outside training budget."""

from src.config.paths import data_path

DATASET_ID = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
RAW_DIRECTORY = data_path("raw/comparison-30m/stories")
DOCUMENT_LIMITS = {"train": 200000, "validation": 1000}
SOURCE_FILES = {"train": "TinyStories-train.txt", "validation": "TinyStories-valid.txt"}
