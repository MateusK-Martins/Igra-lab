"""Tests for turning token streams into language-model examples."""

import pytest
import torch

from src.data.chunker import StoredTokenSequenceDataset, TokenSequenceDataset
from src.data.serialization import TokenStore


def test_creates_shifted_next_token_windows() -> None:
    token_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16], dtype=torch.long)
    dataset = TokenSequenceDataset(token_ids, context_length=3, stride=2)

    assert len(dataset) == 2

    first_input, first_target = dataset[0]
    second_input, second_target = dataset[1]

    torch.testing.assert_close(first_input, torch.tensor([10, 11, 12]))
    torch.testing.assert_close(first_target, torch.tensor([11, 12, 13]))
    torch.testing.assert_close(second_input, torch.tensor([12, 13, 14]))
    torch.testing.assert_close(second_target, torch.tensor([13, 14, 15]))


def test_returns_long_tensors() -> None:
    token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dataset = TokenSequenceDataset(token_ids, context_length=3, stride=1)

    input_ids, target_ids = dataset[0]

    assert input_ids.dtype is torch.long
    assert target_ids.dtype is torch.long


def test_rejects_too_few_tokens() -> None:
    token_ids = torch.tensor([0, 1, 2], dtype=torch.long)

    with pytest.raises(ValueError, match=r"context_length \+ 1"):
        TokenSequenceDataset(token_ids, context_length=3, stride=1)


def test_rejects_non_long_token_ids() -> None:
    token_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    with pytest.raises(ValueError, match="torch.long"):
        TokenSequenceDataset(token_ids, context_length=3, stride=1)


def test_stored_windows_match_memory_windows_and_batch_correctly(tmp_path) -> None:
    from torch.utils.data import DataLoader

    ids = [10, 11, 12, 13, 14, 15, 16]
    store = TokenStore(tmp_path)
    store.write_documents([("a", ids)], tokenizer_sha256="test")
    stored = StoredTokenSequenceDataset(store, context_length=3, stride=2)
    memory = TokenSequenceDataset(torch.tensor(ids), context_length=3, stride=2)
    assert len(stored) == len(memory) == 2
    for index in range(len(stored)):
        for actual, expected in zip(stored[index], memory[index], strict=True):
            torch.testing.assert_close(actual, expected)
            assert actual.dtype == torch.long
    inputs, targets = next(iter(DataLoader(stored, batch_size=2)))
    torch.testing.assert_close(inputs, torch.tensor([[10, 11, 12], [12, 13, 14]]))
    torch.testing.assert_close(targets, torch.tensor([[11, 12, 13], [13, 14, 15]]))
    for index in (-1, len(stored)):
        with pytest.raises(IndexError):
            stored[index]


@pytest.mark.parametrize("ids", [[], [1, 2, 3]])
def test_stored_dataset_rejects_short_stream(tmp_path, ids) -> None:
    store = TokenStore(tmp_path)
    store.write_documents([("a", ids)], tokenizer_sha256="test")
    with pytest.raises(ValueError, match=r"context_length \+ 1"):
        StoredTokenSequenceDataset(store, context_length=3, stride=1)
