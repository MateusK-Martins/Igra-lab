from configs.data.corpus import DATASET_ID, RAW_DIRECTORY, REVISION
from src.data.pipeline import DataSet, DataSetPipe
from src.data.schemas import DocumentManifest
from src.data.sources import JsonlSource

dataset_pipe = DataSetPipe(
    datasets=[
        DataSet(
            manifest=DocumentManifest(
                name=f"tinystories-{REVISION}",
                language="en",
                text_field="text",
                split=split,
                license="CDLA-Sharing-1.0",
                citation=f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}",
            ),
            source=JsonlSource(RAW_DIRECTORY / f"{split}.jsonl"),
        )
        for split in ("train", "validation")
    ]
)
