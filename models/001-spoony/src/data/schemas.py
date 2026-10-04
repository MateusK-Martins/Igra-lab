"""Validated document records and interpretation metadata."""

from dataclasses import dataclass
from typing import Literal

Split = Literal["train", "validation"]


def require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True)
class Document:
    text: str
    source_id: str
    split: Split
    document_id: str

    def __post_init__(self) -> None:
        for field in ("text", "source_id", "document_id"):
            require_text(getattr(self, field), field)
        if self.split not in ("train", "validation"):
            raise ValueError("Document split must be 'train' or 'validation'")


@dataclass(frozen=True)
class DocumentManifest:
    """Name identifies a corpus version; splits may share its name."""

    name: str
    language: str
    text_field: str
    split: Split
    license: str
    citation: str

    def __post_init__(self) -> None:
        for field in ("name", "language", "text_field", "license", "citation"):
            require_text(getattr(self, field), field)
        if self.split not in ("train", "validation"):
            raise ValueError("Manifest split must be 'train' or 'validation'")
