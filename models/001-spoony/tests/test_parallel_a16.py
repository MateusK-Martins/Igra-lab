import math
from types import SimpleNamespace

import pytest
import torch

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import ParallelA16


def test_parallel_outputs_match_two_independent_standard_a16_modules():
    definition = ParallelA16(24, num_heads=2, rope_base=10000.0, residual=True)
    module = build_block(definition, 3).double().eval()
    x = torch.randn(2, 7, 24, dtype=torch.float64, requires_grad=True)
    expected_heads = torch.stack([head(x) for head in module.branch.heads])
    expected = x + expected_heads.sum(0) / math.sqrt(2)
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    parameters = (x, *module.parameters())
    actual_grads = torch.autograd.grad(
        actual.square().sum(), parameters, retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-11, rtol=1e-10)


def test_parallel_a16_cache_matches_full_sequence_and_stays_constant_per_token(
    tmp_path,
):
    definition = ParallelA16(24, num_heads=2, rope_base=10000.0, residual=True)
    module = build_block(definition, 1).double().eval()
    cache = CacheStorage(
        [definition.cache(1)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 13, 24, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat(
            [
                module(x[:, :4], cache=cache),
                module(x[:, 4:9], cache=cache),
                module(x[:, 9:], cache=cache),
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        next_x = torch.randn(2, 1, 24, dtype=torch.float64)
        torch.testing.assert_close(
            module(next_x, cache=cache), module(next_x, cache=restored)
        )
    entry = cache.entries[module.branch.layer_id]
    assert entry.count == 14
    assert entry.keys.shape == (2, 2, 14, 16)
    assert entry.values.shape == (2, 2, 14, 16)
    assert entry.keys.numel() + entry.values.numel() == 2 * 2 * 14 * 32


def test_three_layer_two_head_model_is_near_14m_with_six_independent_a16s():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_912_320
    assert len(model.body) == 12
    attention_blocks = [model.body[index].branch for index in (1, 5, 9)]
    assert all(len(block.heads) == 2 for block in attention_blocks)
    assert len({id(head) for block in attention_blocks for head in block.heads}) == 6
    assert len(entries) == 3
    assert all(entry.keys.shape == (0, 2, 0, 16) for entry in entries)


@pytest.mark.parametrize("heads", [0, -1, True])
def test_parallel_a16_rejects_invalid_head_count(heads):
    with pytest.raises(ValueError, match="num_heads"):
        ParallelA16(384, num_heads=heads)
