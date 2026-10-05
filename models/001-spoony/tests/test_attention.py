import math
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from src.model.assembler import Model
from src.model.cache_storage import CacheEntry, CacheStorage
from src.model.composition import build_block
from src.model.definitions import (
    Embedding,
    GQAAttention,
    LMHead,
    Repeat,
    RMSNorm,
    Sequential,
    SwiGLU,
)
from src.model.position import rope


def definition(**changes):
    args = {
        "input_features": 6,
        "output_features": 5,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "head_features": 4,
        "window_size": 0,
        "dropout": 0.0,
        "position_rotation": partial(rope, base=10000.0),
    }
    args.update(changes)
    return GQAAttention(**args)


def storage_for(
    recipe, capacity=None, dtype=torch.float64, storage_device="cpu", read_device="cpu"
):
    entry = recipe.cache(0)
    if capacity is not None:
        shape = (0, recipe.num_kv_heads, capacity, recipe.head_features)
        entry = CacheEntry(
            entry.layer_id,
            True,
            torch.empty(shape, dtype=dtype),
            torch.empty(shape, dtype=dtype),
            0,
            0,
        )
    return CacheStorage(
        [entry], torch.device(storage_device), torch.device(read_device)
    )


def reference(module, x):
    # Explicit score calculation and per-query indexing, without SDPA or its mask.
    b, t, _ = x.shape
    q = (
        module.q_proj(x)
        .reshape(b, t, module.num_query_heads, module.head_features)
        .transpose(1, 2)
    )
    k = (
        module.k_proj(x)
        .reshape(b, t, module.num_kv_heads, module.head_features)
        .transpose(1, 2)
    )
    v = (
        module.v_proj(x)
        .reshape(b, t, module.num_kv_heads, module.head_features)
        .transpose(1, 2)
    )
    positions = torch.arange(t, device=x.device)
    q, k = (
        module.position_rotation(q, positions),
        module.position_rotation(k, positions),
    )
    outputs = []
    for token in range(t):
        start = max(0, token - module.window_size + 1) if module.window_size else 0
        heads = []
        for head in range(module.num_query_heads):
            kv_head = head // (module.num_query_heads // module.num_kv_heads)
            scores = (
                q[:, head, token : token + 1]
                @ k[:, kv_head, start : token + 1].transpose(-1, -2)
            ) / math.sqrt(module.head_features)
            heads.append(
                (scores.softmax(-1) @ v[:, kv_head, start : token + 1]).squeeze(1)
            )
        outputs.append(torch.cat(heads, dim=-1))
    return module.out_proj(torch.stack(outputs, dim=1))


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("window", [0, 1, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_attention_matches_explicit_reference_and_backward(kv_heads, window, bias):
    torch.manual_seed(7)
    module = (
        definition(num_kv_heads=kv_heads, window_size=window, bias=bias)
        .build()
        .double()
    )
    x = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    actual, expected = module(x), reference(module, x)
    assert actual.shape == (2, 5, 5)
    torch.testing.assert_close(actual, expected)
    variables = (x, *module.parameters())
    grads = torch.autograd.grad(actual.square().sum(), variables)
    reference_grads = torch.autograd.grad(expected.square().sum(), variables)
    for actual_grad, expected_grad in zip(grads, reference_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.isfinite(actual_grad).all()


@pytest.mark.parametrize("window", [0, 3])
def test_future_tokens_cannot_change_earlier_outputs(window):
    module = definition(window_size=window).build().double()
    x = torch.randn(2, 7, 6, dtype=torch.float64)
    altered = x.clone()
    altered[:, 4:] += 100
    torch.testing.assert_close(module(x)[:, :4], module(altered)[:, :4])


def test_window_boundary_includes_exactly_three_tokens():
    module = definition(window_size=3).build().double()
    x = torch.randn(1, 7, 6, dtype=torch.float64)
    altered = x.clone()
    altered[:, :3] += 100
    torch.testing.assert_close(module(x)[:, 5], module(altered)[:, 5])
    altered = x.clone()
    altered[:, 3] += 100
    assert not torch.allclose(module(x)[:, 5], module(altered)[:, 5])


@pytest.mark.parametrize("window", [0, 1, 3])
@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1), (2, 3, 1)])
def test_unbounded_cached_chunks_match_full_forward_and_snapshot(
    tmp_path, window, chunks
):
    recipe = definition(window_size=window)
    module = build_block(recipe, 0).double().eval()
    cache = storage_for(recipe)
    x = torch.randn(2, 6, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        parts, start = [], 0
        for length in chunks:
            parts.append(module(x[:, start : start + length], cache=cache))
            start += length
            assert cache.step(module.layer_id) == start
            path = tmp_path / "cache.pt"
            cache.save(path)
            cache = CacheStorage.load(
                path,
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(parts, dim=1), expected)
    projected = module.k_proj(x).reshape(2, 6, 2, 4).transpose(1, 2)
    torch.testing.assert_close(
        cache.read_all(module.layer_id).keys,
        rope(projected, torch.arange(6), base=10000.0),
    )
    values = module.v_proj(x).reshape(2, 6, 2, 4).transpose(1, 2)
    torch.testing.assert_close(cache.read_all(module.layer_id).values, values)


def test_bounded_single_token_decoding_matches_windowed_attention_after_wrap():
    recipe = definition(window_size=3)
    module = build_block(recipe, 0).double().eval()
    cache = storage_for(recipe, capacity=3)
    x = torch.randn(2, 9, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat(
            [module(x[:, i : i + 1], cache=cache) for i in range(9)], dim=1
        )
    torch.testing.assert_close(actual, expected)
    entry = cache.read_all(module.layer_id)
    assert entry.count == 9
    assert entry.keys.shape[2] == 3


def test_bounded_oversized_chunk_uses_only_post_write_retained_keys():
    # Uniform scores make the accepted write-first semantics independently visible.
    recipe = definition(
        input_features=2,
        output_features=2,
        num_query_heads=1,
        num_kv_heads=1,
        head_features=2,
        window_size=3,
    )
    module = build_block(recipe, 0).double()
    with torch.no_grad():
        module.q_proj.weight.zero_()
        module.k_proj.weight.zero_()
        module.v_proj.weight.copy_(torch.eye(2))
        module.out_proj.weight.copy_(torch.eye(2))
    cache = storage_for(recipe, capacity=3)
    x = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2)
    with torch.no_grad():
        actual = module(x, cache=cache)
    # Retained positions [3,4,5]; queries [0,1,2] have no allowed keys.
    expected = torch.stack(
        [
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            x[0, 3],
            x[0, 3:5].mean(0),
            x[0, 3:6].mean(0),
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(actual, expected)
    assert cache.step(module.layer_id) == 6


def test_dropout_disabled_in_eval_and_passed_in_training():
    module = definition(dropout=0.5).build()
    x = torch.randn(1, 4, 6)
    original = F.scaled_dot_product_attention
    with patch(
        "src.model.attention.F.scaled_dot_product_attention", wraps=original
    ) as sdpa:
        module.eval()
        first, second = module(x), module(x)
        torch.testing.assert_close(first, second)
        assert sdpa.call_args.kwargs["dropout_p"] == 0.0
        module.train()
        module(x)
        assert sdpa.call_args.kwargs["dropout_p"] == 0.5


def test_full_model_real_attention_cache_and_gradients():
    attention = definition(input_features=4, output_features=4, residual=True)
    recipe = Model(
        SimpleNamespace(vocab_size=11),
        Embedding(4),
        [
            Repeat(
                2,
                [
                    Sequential(
                        [RMSNorm(4, 1e-6), attention, SwiGLU(4, 8, 4, residual=True)]
                    )
                ],
            )
        ],
        LMHead(4, True),
    )
    model, entries = recipe.assemble()
    model.double().eval()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits = model(tokens)
    assert logits.shape == (2, 4, 11)
    logits.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        cached = torch.cat(
            [model(tokens[:, :2], cache=cache), model(tokens[:, 2:], cache=cache)],
            dim=1,
        )
    torch.testing.assert_close(cached, logits.detach())
    assert len(entries) == 2
    assert all(e.count == 4 for e in entries)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_query_heads", 0),
        ("num_kv_heads", 0),
        ("head_features", True),
        ("window_size", -1),
        ("window_size", True),
        ("dropout", 1.0),
        ("dropout", float("nan")),
        ("bias", 1),
        ("residual", 1),
    ],
)
def test_definition_rejects_invalid_attention_settings(field, value):
    with pytest.raises(ValueError, match=field):
        definition(**{field: value})


def test_definition_rejects_nondivisible_head_counts():
    with pytest.raises(ValueError, match="divisible"):
        definition(num_query_heads=3, num_kv_heads=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_attention_with_cpu_storage_matches_uncached_forward():
    recipe = definition(window_size=3)
    module = build_block(recipe, 0).cuda().eval()
    cache = storage_for(recipe, storage_device="cpu", read_device="cuda")
    x = torch.randn(2, 6, 6, device="cuda")
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat(
            [module(x[:, :2], cache=cache), module(x[:, 2:], cache=cache)], dim=1
        )
    torch.testing.assert_close(actual, expected)
    assert cache.entries[module.layer_id].keys.device.type == "cpu"
