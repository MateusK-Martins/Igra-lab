from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest

from src.data.pipeline import DataSet, DataSetPipe
from src.data.schemas import Document, DocumentManifest


@dataclass
class FakeSource:
    rows: list[Mapping[str, object]]
    reads: int = 0

    def read(self) -> Iterator[Mapping[str, object]]:
        self.reads += 1
        yield from self.rows


def make_dataset(name, split, rows):
    return DataSet(
        manifest=DocumentManifest(name, "en", "story", split, "test", "test"),
        source=FakeSource(rows),
    )


def test_pipe_filters_preserves_order_and_reopens_sources() -> None:
    first = make_dataset("first", "train", [{"story": "First."}])
    validation = make_dataset("first", "validation", [{"story": "Held out."}])
    second = make_dataset("second", "train", [{"story": "Second."}])
    pipe = DataSetPipe([first, validation, second])
    traversal = pipe.documents("train")
    assert first.source.reads == 0
    expected = [
        Document("First.", "first", "train", "first/train/0"),
        Document("Second.", "second", "train", "second/train/0"),
    ]
    assert list(traversal) == expected
    assert list(pipe.documents("train")) == expected
    assert first.source.reads == 2
    assert validation.source.reads == 0
    assert next(pipe.documents("validation")).text == "Held out."


@pytest.mark.parametrize("row", [{}, {"story": 42}, {"story": " "}])
def test_bad_text_fails_before_delivery(row) -> None:
    with pytest.raises((ValueError, TypeError)):
        list(make_dataset("bad", "train", [row]).documents())


def test_duplicate_identity_is_rejected() -> None:
    dataset = make_dataset("duplicate", "train", [])
    with pytest.raises(ValueError, match="unique"):
        DataSetPipe([dataset, dataset])


@pytest.mark.parametrize("split", ["", "test", " train "])
def test_document_rejects_invalid_split(split) -> None:
    with pytest.raises(ValueError, match="split"):
        Document("Text", "source", split, "id")


def test_empty_pipe_and_invalid_requested_split() -> None:
    pipe = DataSetPipe([])
    assert list(pipe.documents("train")) == []
    with pytest.raises(ValueError, match="split"):
        list(pipe.documents("test"))
