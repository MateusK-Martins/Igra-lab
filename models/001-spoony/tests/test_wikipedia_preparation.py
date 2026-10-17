"""Verify exact budgets, content deduplication and train-only vocabulary input."""

import json

import pytest

from src.data import wikipedia as preparation
from src.data.serialization import TokenStore


def test_slice_writer_caps_tokens_and_preserves_offsets(tmp_path):
    writer = preparation.SliceWriter(tmp_path / "split", tmp_path / "raw.jsonl", 5)
    writer.append({"id": "1", "text": "first"}, [1, 2, 3], "a")
    writer.append({"id": "2", "text": "second"}, [4, 5, 6], "b")
    writer.append({"id": "3", "text": "unused"}, [7], "c")
    writer.close("hash")
    store = TokenStore(tmp_path / "split")
    assert store.read_summary().token_count == 5
    assert store.read_tokens(0, 5) == [1, 2, 3, 4, 5]
    assert [(o.start, o.end) for o in store.iter_document_offsets()] == [(0, 3), (3, 5)]
    assert writer.truncated_final_document
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 2


def test_preparation_exact_caps_disjoint_splits_and_train_only_tokenizer(
    tmp_path, monkeypatch
):
    rows = [
        {
            "id": str(i),
            "text": f"An article about trees, animals and rivers number {i}.",
        }
        for i in range(1000)
    ]
    training = [
        r
        for r in rows
        if preparation.split_for(preparation.content_hash(r["text"])) == "train"
    ]
    validation = [
        r
        for r in rows
        if preparation.split_for(preparation.content_hash(r["text"])) == "validation"
    ]
    selected = [
        training[0],
        training[0],
        validation[0],
        *training[1:8],
        *validation[1:4],
    ]
    output = tmp_path / "processed"
    monkeypatch.setattr(preparation, "RAW_DIRECTORY", tmp_path / "raw")
    monkeypatch.setattr(
        preparation,
        "preparation_config",
        type(preparation.preparation_config)(
            tokenizer=preparation.preparation_config.tokenizer,
            output_directory=output,
            end_of_document_token=preparation.preparation_config.end_of_document_token,
        ),
    )
    monkeypatch.setattr(preparation, "TRAIN_TOKENS", 80)
    monkeypatch.setattr(preparation, "VALIDATION_TOKENS", 20)
    monkeypatch.setattr(preparation, "articles", lambda: iter(selected))

    def forbid_vocabulary_training(*args, **kwargs):
        raise AssertionError("Cloud preparation must reuse the bundled tokenizer")

    monkeypatch.setattr(
        preparation.ByteLevelBPETokenizer, "train", forbid_vocabulary_training
    )

    def interrupted_articles():
        yield selected[0]
        raise RuntimeError("interrupted preparation")

    monkeypatch.setattr(preparation, "articles", interrupted_articles)
    with pytest.raises(RuntimeError, match="interrupted preparation"):
        preparation.main()
    assert not output.exists()
    assert output.with_name(output.name + ".preparing").exists()
    monkeypatch.setattr(preparation, "articles", lambda: iter(selected))
    preparation.main()
    assert (
        output / "tokenizer.json"
    ).read_bytes() == preparation.TOKENIZER_ARTIFACT.read_bytes()
    assert TokenStore(output / "train").read_summary().token_count == 80
    assert TokenStore(output / "validation").read_summary().token_count == 20
    sets = {}
    for split in ("train", "validation"):
        records = [
            json.loads(line)
            for line in (tmp_path / "raw" / f"{split}.jsonl").read_text().splitlines()
        ]
        hashes = [r["content_sha256"] for r in records]
        assert len(hashes) == len(set(hashes))
        assert all(preparation.split_for(h) == split for h in hashes)
        sets[split] = set(hashes)
        list(TokenStore(output / split).iter_document_offsets())
    assert sets["train"].isdisjoint(sets["validation"])
    assert (
        json.loads((output / "preparation-report.json").read_text())[
            "duplicates_skipped"
        ]
        == 1
    )
    with pytest.raises(FileExistsError):
        preparation.main()
