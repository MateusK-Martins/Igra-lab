"""Compose sources into ordered, repeatable document traversals."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from src.data.schemas import Document, DocumentManifest, Split
from src.data.sources import Source


@dataclass(frozen=True)
class DataSet:
    manifest: DocumentManifest
    source: Source

    def documents(self) -> Iterator[Document]:
        for index, row in enumerate(self.source.read()):
            identity = f"{self.manifest.name}/{self.manifest.split}/{index}"
            if not isinstance(row, Mapping):
                raise TypeError(f"Record {identity} must be a mapping")
            if self.manifest.text_field not in row:
                raise ValueError(
                    f"Record {identity} is missing text field {self.manifest.text_field!r}"
                )
            text = row[self.manifest.text_field]
            if not isinstance(text, str):
                raise TypeError(f"Record {identity} text must be a string")
            yield Document(text, self.manifest.name, self.manifest.split, identity)


@dataclass
class DataSetPipe:
    datasets: list[DataSet]

    def __post_init__(self) -> None:
        self._validate_identities()

    def _validate_identities(self) -> None:
        identities = [
            (item.manifest.name, item.manifest.split) for item in self.datasets
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("Dataset names must be unique within each split")

    def documents(self, split: Split) -> Iterator[Document]:
        if split not in ("train", "validation"):
            raise ValueError("Requested split must be 'train' or 'validation'")
        self._validate_identities()
        for dataset in self.datasets:
            if dataset.manifest.split == split:
                yield from dataset.documents()
