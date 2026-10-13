import io
import json

import pytest

from scripts import materialize_corpus
from src.data.sources import JsonlSource


def test_materialization_is_bounded_and_excludes_cross_split_duplicates(
    tmp_path, monkeypatch
):
    destination = tmp_path / "subset"
    monkeypatch.setattr(materialize_corpus, "RAW_DIRECTORY", destination)
    monkeypatch.setattr(
        materialize_corpus, "DOCUMENT_LIMITS", {"train": 2, "validation": 1}
    )
    payloads = iter(
        [
            b"Alpha\n<|endoftext|>\nAlpha\n<|endoftext|>\nBeta\n<|endoftext|>\nUnused\n",
            b"Alpha\n<|endoftext|>\nGamma\n<|endoftext|>\nUnused\n",
        ]
    )
    monkeypatch.setattr(
        materialize_corpus,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(next(payloads)),
    )
    materialize_corpus.main()
    train = JsonlSource(destination / "train.jsonl")
    assert list(train.read()) == [{"text": "Alpha"}, {"text": "Beta"}]
    assert list(train.read()) == list(train.read())
    assert list(JsonlSource(destination / "validation.jsonl").read()) == [
        {"text": "Gamma"}
    ]
    report = json.loads((destination / "subset.json").read_text())
    assert report["splits"]["train"]["duplicates_or_empty_skipped"] == 1
    assert report["splits"]["validation"]["duplicates_or_empty_skipped"] == 1
    with pytest.raises(FileExistsError):
        materialize_corpus.main()


def test_materialization_rejects_incomplete_source(tmp_path, monkeypatch):
    monkeypatch.setattr(materialize_corpus, "RAW_DIRECTORY", tmp_path / "subset")
    monkeypatch.setattr(materialize_corpus, "DOCUMENT_LIMITS", {"train": 2})
    monkeypatch.setattr(
        materialize_corpus,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"Alpha\n<|endoftext|>\n"),
    )
    with pytest.raises(ValueError, match="1/2"):
        materialize_corpus.main()
