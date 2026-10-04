import hashlib
import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from src.config.types import TokenizerConfig
from src.data.pipeline import DataSet, DataSetPipe
from src.data.prepare import DataProcessor, train_tokenizer
from src.data.schemas import DocumentManifest
from src.data.serialization import DocumentOffset, TokenStore
from src.data.tokenizer import ByteLevelBPETokenizer


@dataclass
class FakeSource:
    texts: tuple[str, ...]

    def read(self):
        for text in self.texts:
            yield {"text": text}


@pytest.fixture
def processor(tmp_path):
    train_texts = ("A little fox found a bird.", "The bird flew home.")
    tokenizer = ByteLevelBPETokenizer.train(
        train_texts,
        target_vocab_size=300,
        min_frequency=1,
        special_tokens=("<|unk|>", "<|eot|>"),
    )
    pipe = DataSetPipe(
        [
            DataSet(
                DocumentManifest("test", "en", "text", "train", "test", "test"),
                FakeSource(train_texts),
            ),
            DataSet(
                DocumentManifest("test", "en", "text", "validation", "test", "test"),
                FakeSource(("An unseen story about a fox.",)),
            ),
        ]
    )
    return DataProcessor(
        tokenizer,
        pipe,
        tmp_path / "processed",
        end_of_document_token="<|eot|>",
    )


def test_first_run_writes_exact_tokens_offsets_and_hash(processor) -> None:
    summary = processor.process_split("train")
    store = TokenStore(processor.output_directory / "train")
    expected = []
    offsets = []
    for document in processor.dataset_pipe.documents("train"):
        start = len(expected)
        expected.extend(processor.tokenizer.encode(document.text))
        expected.append(processor.end_token_id)
        offsets.append(DocumentOffset(document.document_id, start, len(expected)))
    assert summary.document_count == 2
    assert summary.token_count == len(expected)
    assert store.read_tokens(0, summary.token_count) == expected
    assert list(store.iter_document_offsets()) == offsets
    tokenizer_path = processor.output_directory / "tokenizer.json"
    restored = ByteLevelBPETokenizer.load(tokenizer_path)
    assert restored.encode("A little fox.") == processor.tokenizer.encode(
        "A little fox."
    )
    metadata = json.loads(store.metadata_path.read_text())
    assert (
        metadata["tokenizer_sha256"]
        == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    )


def test_validation_reuses_saved_tokenizer_without_overwriting(processor) -> None:
    processor.process_split("train")
    tokenizer_path = processor.output_directory / "tokenizer.json"
    before = tokenizer_path.read_bytes()
    original_save = processor.tokenizer.save
    with patch.object(processor.tokenizer, "save", wraps=original_save) as save:
        summary = processor.process_split("validation")
        assert all(call.args[0] != tokenizer_path for call in save.call_args_list)
    assert tokenizer_path.read_bytes() == before
    expected = processor.tokenizer.encode("An unseen story about a fox.") + [
        processor.end_token_id
    ]
    store = TokenStore(processor.output_directory / "validation")
    assert summary.document_count == 1
    assert store.read_tokens(0, summary.token_count) == expected


def test_different_tokenizer_is_rejected_without_changing_artifacts(processor) -> None:
    processor.process_split("train")
    before = {
        path.relative_to(processor.output_directory): path.read_bytes()
        for path in processor.output_directory.rglob("*")
        if path.is_file()
    }
    other = ByteLevelBPETokenizer.train(
        ["Different vocabulary different vocabulary."],
        target_vocab_size=300,
        min_frequency=1,
        special_tokens=("<|unk|>", "<|eot|>"),
    )
    replacement = DataProcessor(
        other,
        processor.dataset_pipe,
        processor.output_directory,
        end_of_document_token="<|eot|>",
    )
    with pytest.raises(ValueError, match="different tokenizer"):
        replacement.process_split("validation")
    after = {
        path.relative_to(processor.output_directory): path.read_bytes()
        for path in processor.output_directory.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_repeated_split_is_rejected(processor) -> None:
    processor.process_split("train")
    with pytest.raises(FileExistsError):
        processor.process_split("train")


def test_invalid_split_creates_no_files(processor) -> None:
    with pytest.raises(ValueError, match="Split"):
        processor.process_split("test")
    assert not processor.output_directory.exists()


def test_train_tokenizer_consumes_only_train_and_pipe_can_be_reused(processor) -> None:
    config = TokenizerConfig(300, 1, ("<|unk|>", "<|eot|>"))
    training_source = processor.dataset_pipe.datasets[0].source
    validation_source = processor.dataset_pipe.datasets[1].source
    real_train = ByteLevelBPETokenizer.train
    received_texts = []

    def capture_training(texts, **settings):
        received_texts.extend(texts)
        return real_train(received_texts, **settings)

    with (
        patch.object(training_source, "read", wraps=training_source.read) as train_read,
        patch.object(
            validation_source, "read", wraps=validation_source.read
        ) as valid_read,
        patch(
            "src.data.prepare.ByteLevelBPETokenizer.train",
            side_effect=capture_training,
        ) as train,
    ):
        tokenizer = train_tokenizer(processor.dataset_pipe, config)
        assert received_texts == list(training_source.texts)
        train_read.assert_called_once()
        valid_read.assert_not_called()
        assert train.call_args.kwargs == {
            "target_vocab_size": config.target_vocab_size,
            "min_frequency": config.min_frequency,
            "special_tokens": config.special_tokens,
        }

        repeated = list(processor.dataset_pipe.documents("train"))
        assert [document.text for document in repeated] == received_texts
        assert train_read.call_count == 2
        valid_read.assert_not_called()

    assert tokenizer.decode(tokenizer.encode(received_texts[0])) == received_texts[0]
