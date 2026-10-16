from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import LatentAttention
from src.model.position import rope


def definition(rope_base=10000.0):
    return LatentAttention(
        12, "sma4_key_gelu16", rope_base=rope_base, matrix_features=8
    )


def reference(m, x):
    q, k = m.q_proj(x), m.k_proj(x)
    matrices = m.matrix_proj(x).reshape(*x.shape[:2], 8, 8)
    if m.rope_base is not None:
        pos = torch.arange(x.shape[1])
        q = rope(q[:, None], pos, base=m.rope_base)[:, 0]
        k = rope(k[:, None], pos, base=m.rope_base)[:, 0]
        matrices = rope(matrices.transpose(1, 2), pos, base=m.rope_base).transpose(1, 2)
    result = []
    for t in range(x.shape[1]):
        responses = (matrices[:, : t + 1] @ q[:, t, :, None][:, None]).squeeze(-1)
        weights = ((k[:, : t + 1] * q[:, t, None]).sum(-1) / 8**0.5).softmax(-1)
        responses = F.gelu(m.response_proj(responses))
        result.append(m.out_proj((responses * weights[..., None]).sum(1)))
    return torch.stack(result, 1)


@pytest.mark.parametrize("base", [None, 10000.0])
def test_sma8_independent_reference_all_gradients(base):
    m = build_block(definition(base), 0).double()
    x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    actual, expected = m(x), reference(m, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = (x, *m.parameters())
    a = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), params)
    for ga, gb in zip(a, b, strict=True):
        torch.testing.assert_close(ga, gb, atol=1e-11, rtol=1e-10)


def test_sma8_cache_chunks_single_tokens_snapshot_and_dimension_rejection(tmp_path):
    d = definition()
    m = build_block(d, 0).double().eval()
    x = torch.randn(2, 19, 12, dtype=torch.float64)
    cache = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        full = m(x)
        start = m(x[:, :7], cache=cache)
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        end = torch.cat([m(p, cache=restored) for p in x[:, 7:].split(1, 1)], 1)
        torch.testing.assert_close(
            torch.cat([start, end], 1), full, atol=1e-12, rtol=1e-11
        )
    entry = restored.matrices[m.layer_id]
    assert entry.matrices.numel() + entry.keys.numel() == 2 * 19 * 72
    with pytest.raises(ValueError):
        restored.write_matrices(
            m.layer_id, torch.randn(2, 1, 4, 4), keys=torch.randn(2, 1, 4)
        )
    assert entry.count == 19
    old = LatentAttention(12, "sma4_key_gelu16", rope_base=10000.0)
    assert old.cache(0).layer_id != d.cache(0).layer_id
    with pytest.raises(RuntimeError):
        m.load_state_dict(old.build().state_dict())


def test_sma8_causality_and_gradcheck():
    m = build_block(definition(), 0).double()
    x = torch.randn(1, 4, 12, dtype=torch.float64, requires_grad=True)
    out = m(x)
    changed = x.detach().clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(m(changed)[:, :2], out[:, :2], atol=0, rtol=0)
    grad = torch.autograd.grad(out[:, :2].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 2:]) == 0
    assert torch.autograd.gradcheck(m, (x,), fast_mode=True)


def test_sma8_parameter_count_and_shared_initialization():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 10_453_008
    assert len(entries) == 6
    assert all(e.matrices.shape == (0, 0, 8, 8) for e in entries)
    a = definition().build()
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    a = definition().build()
    assert torch.equal(torch.get_rng_state(), before)
    assert a.response_proj.weight.shape == (16, 8)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sma8_bf16_forward_backward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = build_block(definition(), 0).to(device)
    x = torch.randn(2, 257, 12, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = m(x)
        loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("features", [0, -2, 3, True])
def test_sma8_invalid_dimensions(features):
    with pytest.raises(ValueError):
        LatentAttention(
            12, "sma4_key_gelu16", rope_base=10000.0, matrix_features=features
        )
