from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import ACSwiGLU
from src.model.gmlp import GMLPImpl


def definition():
    return ACSwiGLU(8, 16, gmlp_hidden_features=24, gmlp_window_size=10)


def gmlp_reference(m, x):
    u, v = F.gelu(m.up(x)).chunk(2, -1)
    v = m.norm(v)
    result = []
    for t in range(x.shape[1]):
        row = t % m.window_size
        start = t - row
        mix = m.spatial_bias[row] + sum(
            m.spatial_weight[row, j] * v[:, start + j] for j in range(row + 1)
        )
        result.append(m.down(u[:, t] * mix))
    return torch.stack(result, 1)


def test_gmlp_ten_token_matrix_reference_and_all_gradients():
    m = GMLPImpl(8, 24, 10).double()
    x = torch.randn(2, 23, 8, dtype=torch.float64, requires_grad=True)
    actual, expected = m(x), gmlp_reference(m, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = (x, *m.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    assert torch.count_nonzero(ga[1].triu(1)) == 0  # spatial matrix is causal


def test_full_width_order_raw_attention_condition_and_exact_residuals():
    m = build_block(definition(), 0).double()
    with torch.no_grad():
        m.condition.weight.normal_(std=0.1)
    x = torch.randn(2, 23, 8, dtype=torch.float64, requires_grad=True)
    z = m.attention.retrieve(m.attention_norm(x))
    r = x + m.attention.out_proj(z)
    r = r + gmlp_reference(m.gmlp, m.gmlp_norm(r))
    u = m.ffn_norm(r)
    expected = r + m.ffn.down((F.silu(m.ffn.gate(u)) * m.ffn.up(u)) * m.modulation(z))
    actual = m(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    params = (x, *m.parameters())
    ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    assert m.gmlp.up.in_features == 8


def test_chunks_boundaries_snapshot_and_constant_gmlp_cache(tmp_path):
    d = definition()
    m = build_block(d, 0).double().eval()
    cache = CacheStorage(d.cache(0), torch.device("cpu"), torch.device("cpu"))
    x = torch.randn(2, 57, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = m(x)
        first = m(x[:, :17], cache=cache)
        path = tmp_path / "snapshot.pt"
        cache.save(path)
        cache = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        actual = torch.cat(
            [first, *[m(part, cache=cache) for part in x[:, 17:].split(1, 1)]], 1
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        entry = cache.entries[m.gmlp.layer_id]
        assert entry.keys.shape == (2, 1, 10, 12)
        assert entry.count == 57
        m(torch.randn(2, 101, 8, dtype=torch.float64), cache=cache)
        assert entry.keys.shape == (2, 1, 10, 12) and entry.count == 158
    with pytest.raises(RuntimeError):
        m(x[:, :1], cache=cache)


def test_causality_window_reset_and_future_gradient_isolation():
    m = build_block(definition(), 0).double()
    x = torch.randn(1, 23, 8, dtype=torch.float64, requires_grad=True)
    output = m(x)
    changed = x.detach().clone()
    changed[:, 13:] += 100
    torch.testing.assert_close(m(changed)[:, :13], output[:, :13], atol=0, rtol=0)
    grad = torch.autograd.grad(output[:, :13].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 13:]) == 0
    branch = m.gmlp(x)
    old_changed = x.detach().clone()
    old_changed[:, :10] += 100
    torch.testing.assert_close(
        m.gmlp(old_changed)[:, 10:], branch[:, 10:], atol=0, rtol=0
    )
    assert torch.autograd.gradcheck(
        m, (x[:, :3].detach().requires_grad_(),), fast_mode=True
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_finite_forward_all_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = build_block(definition(), 0).to(device)
    x = torch.randn(2, 257, 8, device=device, requires_grad=True)
    with torch.autocast(device, dtype=torch.bfloat16):
        output = m(x)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_actual_parameters_full_width_and_unchanged_initial_weights(monkeypatch):
    import configs.model.model as config

    torch.manual_seed(42)
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 14_032_788
    assert len(entries) == 12
    assert all(
        m.gmlp.up.in_features == 384 and m.gmlp.spatial_weight.shape == (10, 10)
        for m in model.body[::2]
    )
    torch.manual_seed(42)
    monkeypatch.setattr(config, "GMLP_HIDDEN_FEATURES", 0)
    original, _ = build_model(SimpleNamespace(vocab_size=8192))
    new = model.state_dict()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(new[name], value, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gmlp_hidden_features": -2},
        {"gmlp_hidden_features": 3},
        {"gmlp_hidden_features": True},
        {"gmlp_window_size": 0},
    ],
)
def test_invalid_gmlp_settings(kwargs):
    with pytest.raises(ValueError):
        ACSwiGLU(8, **kwargs)
