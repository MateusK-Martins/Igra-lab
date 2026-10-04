from dataclasses import replace

import pytest
import torch

from src.config.types import DataLoaderConfig
from src.data.chunker import StoredTokenSequenceDataset
from src.data.dataloader import build_dataloader
from src.data.serialization import TokenStore


@pytest.fixture
def dataset(tmp_path):
    store = TokenStore(tmp_path)
    store.write_documents([("test", list(range(41)))], tokenizer_sha256="test")
    return StoredTokenSequenceDataset(store, context_length=4, stride=4)


def config(**overrides):
    return replace(DataLoaderConfig(3, False, 0, False, False, 42), **overrides)


def test_batches_preserve_targets_and_keep_partial_batch(dataset) -> None:
    batches = list(build_dataloader(dataset, config()))
    assert [inputs.shape[0] for inputs, _ in batches] == [3, 3, 3, 1]
    for inputs, targets in batches:
        assert inputs.shape == targets.shape
        assert inputs.shape[1] == 4
        assert inputs.dtype == targets.dtype == torch.long
        torch.testing.assert_close(targets, inputs + 1)
    torch.testing.assert_close(batches[0][0], torch.arange(12).reshape(3, 4))


def test_drop_last_discards_partial_batch(dataset) -> None:
    batches = list(build_dataloader(dataset, config(drop_last=True)))
    assert len(batches) == 3
    assert all(inputs.shape == (3, 4) for inputs, _ in batches)


def test_shuffle_is_repeatable_and_keeps_all_examples(dataset) -> None:
    settings = config(shuffle=True)
    first = torch.cat([inputs for inputs, _ in build_dataloader(dataset, settings)])
    second = torch.cat([inputs for inputs, _ in build_dataloader(dataset, settings)])
    torch.testing.assert_close(first, second)
    assert sorted(first[:, 0].tolist()) == list(range(0, 40, 4))
    assert first[:, 0].tolist() != list(range(0, 40, 4))
