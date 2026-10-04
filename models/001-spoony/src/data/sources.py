"""Sources supply raw records through a fresh read on every traversal."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from datasets import load_dataset

from src.data.schemas import require_text


class Source(Protocol):
    """Repeatable source with stable order for an unchanged version."""

    def read(self) -> Iterator[Mapping[str, object]]: ...


@dataclass(frozen=True)
class HuggingFaceSource:
    dataset_id: str
    revision: str
    split: str
    streaming: bool = False

    def __post_init__(self) -> None:
        for field in ("dataset_id", "revision", "split"):
            require_text(getattr(self, field), field)

    def read(self) -> Iterator[Mapping[str, object]]:
        dataset = load_dataset(
            self.dataset_id,
            revision=self.revision,
            split=self.split,
            streaming=self.streaming,
        )
        yield from dataset
