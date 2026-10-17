"""Materialized Wikipedia articles, split by content hash before tokenization."""

from configs.data.wikipedia import DATASET_ID, RAW_DIRECTORY, REVISION, SUBSET
from src.data.pipeline import DataSet, DataSetPipe
from src.data.schemas import DocumentManifest
from src.data.sources import JsonlSource

dataset_pipe = DataSetPipe(
    datasets=[
        DataSet(
            manifest=DocumentManifest(
                name=f"wikipedia-{SUBSET}-{REVISION}",
                language="en",
                text_field="text",
                split=split,
                license="CC-BY-SA-3.0 / GFDL",
                citation=f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}/{SUBSET}",
            ),
            source=JsonlSource(RAW_DIRECTORY / f"{split}.jsonl"),
        )
        for split in ("train", "validation")
    ]
)
