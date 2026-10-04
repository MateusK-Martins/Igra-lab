import json

import pytest

from src.data.serialization import DocumentOffset, TokenStore, TokenStoreSummary


@pytest.fixture
def store(tmp_path):
    store = TokenStore(tmp_path / "train")
    store.write_documents(
        [("a", [42, 81, 1]), ("b", [56, 2**32 - 1, 1])],
        tokenizer_sha256="test-hash",
    )
    return store


def test_round_trip_from_new_store_instance(store) -> None:
    reader = TokenStore(store.directory)
    assert reader.read_summary() == TokenStoreSummary(2, 6)
    assert reader.read_tokens(1, 5) == [81, 1, 56, 2**32 - 1]
    assert reader.read_tokens(6, 6) == []
    assert list(reader.iter_document_offsets()) == [
        DocumentOffset("a", 0, 3),
        DocumentOffset("b", 3, 6),
    ]


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2), (0, 7)])
def test_rejects_out_of_bounds_ranges(store, start, end) -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        store.read_tokens(start, end)


@pytest.mark.parametrize("start,end", [(True, 2), (0, 2.0)])
def test_rejects_non_integer_offsets(store, start, end) -> None:
    with pytest.raises(TypeError):
        store.read_tokens(start, end)


def test_rejects_truncated_tokens(store) -> None:
    store.tokens_path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="size"):
        store.read_tokens(0, 1)


@pytest.mark.parametrize(
    "records",
    [
        [{"document_id": "a", "start": 1, "end": 3}],
        [{"document_id": "a", "start": 0, "end": 7}],
        [{"document_id": "a", "start": 0, "end": 3}],
        [
            {"document_id": "a", "start": 0, "end": 3},
            {"document_id": "b", "start": 3, "end": 5},
        ],
    ],
)
def test_rejects_inconsistent_document_index(store, records) -> None:
    store.documents_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        list(store.iter_document_offsets())


def test_empty_store(tmp_path) -> None:
    store = TokenStore(tmp_path)
    store.write_documents([], tokenizer_sha256="test-hash")
    assert store.read_summary() == TokenStoreSummary(0, 0)
    assert store.read_tokens(0, 0) == []
    assert list(store.iter_document_offsets()) == []
