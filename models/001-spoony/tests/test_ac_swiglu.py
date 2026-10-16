from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from configs.model.model import CONDITIONED, build_model
from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import ACSwiGLU
from src.model.position import rope


def reference(m, x):
    a = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + m.attention_norm.eps)
        * m.attention_norm.weights
    )
    q, k, v = m.attention.q_proj(a), m.attention.k_proj(a), m.attention.v_proj(a)
    pos = torch.arange(x.shape[1], device=x.device)
    q = rope(q[:, None], pos, base=10000.0)[:, 0]
    k = rope(k[:, None], pos, base=10000.0)[:, 0]
    scores = q @ k.transpose(-1, -2) / 4
    mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril()
    z = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
    r = x + F.linear(z, m.attention.out_proj.weight)
    u = (
        r
        * torch.rsqrt(r.square().mean(-1, keepdim=True) + m.ffn_norm.eps)
        * m.ffn_norm.weights
    )
    h = F.silu(F.linear(u, m.ffn.gate.weight)) * F.linear(u, m.ffn.up.weight)
    gamma = 1 + m.alpha * F.linear(z, m.condition.weight).tanh()
    return r + F.linear(h * gamma, m.ffn.down.weight)


def test_formula_and_all_gradients():
    m = ACSwiGLU(4, 8).build().double()
    torch.nn.init.normal_(m.condition.weight, std=0.1)
    other = deepcopy(m)
    x = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    actual = m(x)
    expected = reference(other, xx)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-9)
    assert torch.autograd.gradcheck(m, (x,))


def test_zero_conditioning_equivalence_and_learning_signal():
    torch.manual_seed(42)
    m = ACSwiGLU(4, 8).build().double()
    torch.manual_seed(42)
    control = ACSwiGLU(4, 8, conditioned=False).build().double()
    a = m.state_dict()
    b = control.state_dict()
    for name, value in b.items():
        torch.testing.assert_close(a[name], value, atol=0, rtol=0)
    x = torch.randn(2, 7, 4, dtype=torch.float64)
    torch.testing.assert_close(m(x), control(x), atol=0, rtol=0)
    assert not m.condition.weight.count_nonzero()
    m(x).square().sum().backward()
    assert (
        m.condition.weight.grad.isfinite().all()
        and m.condition.weight.grad.abs().sum() > 0
    )
    assert torch.equal(
        m.modulation(torch.randn(2, 3, 16, dtype=torch.float64)),
        torch.ones(2, 3, 8, dtype=torch.float64),
    )


def test_gamma_bounds_signed_directions():
    m = ACSwiGLU(4, 8).build().double()
    torch.nn.init.normal_(m.condition.weight, std=100.0)
    gamma = m.modulation(torch.randn(2, 7, 16, dtype=torch.float64))
    assert gamma.min() >= 0.9 and gamma.max() <= 1.1
    assert (gamma < 1).any() and (gamma > 1).any()


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_cache_causality_snapshot_and_no_extra_history(chunk, tmp_path):
    d = ACSwiGLU(4, 8)
    m = build_block(d, 0).double()
    torch.nn.init.normal_(m.condition.weight, std=0.1)
    x = torch.randn(2, 17, 4, dtype=torch.float64, requires_grad=True)
    y = m(x)
    (gradient,) = torch.autograd.grad(y[:, :4].sum(), x)
    assert not gradient[:, 4:].count_nonzero()
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        parts = []
        for p in x.split(chunk, 1):
            parts.append(m(p, cache=c))
            c.save(tmp_path / "cache.pt")
            c = CacheStorage.load(
                tmp_path / "cache.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(parts, 1), y, atol=1e-12, rtol=1e-12)
    assert len(c.entries) == 1 and not c.matrices
    e = next(iter(c.entries.values()))
    assert e.keys.numel() + e.values.numel() == 2 * 17 * 32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_gradients_and_equivalence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    m = ACSwiGLU(8, 16).build().to(device)
    torch.manual_seed(42)
    control = ACSwiGLU(8, 16, conditioned=False).build().to(device)
    x = torch.randn(2, 65, 8, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = m(x)
        expected = control(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
    with torch.autocast(device, dtype=torch.bfloat16):
        gamma = m.modulation(torch.randn(2, 3, 16, device=device))
    assert gamma.dtype == torch.float32 and gamma.min() >= 0.9 and gamma.max() <= 1.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_features": 0},
        {"alpha": 0},
        {"alpha": float("nan")},
        {"conditioned": 1},
        {"residual": True},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ACSwiGLU(4, **kwargs)


def test_full_model_count_cache_and_no_persistent_retrieval():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == (
        10_476_288 if CONDITIONED else 10_377_984
    )
    assert len(entries) == 6 and all(e.keys.shape == (0, 1, 0, 16) for e in entries)
    assert all(
        (layer.condition is not None and not layer.condition.weight.count_nonzero())
        if CONDITIONED
        else layer.condition is None
        for layer in model.body[::2]
    )
    assert not any(
        "retrieval" in name or "gamma" in name for name, _ in model.named_buffers()
    )


def test_matched_models_share_initial_parameters_outputs_and_rng(monkeypatch):
    import configs.model.model as config

    models = []
    for enabled in (True, False):
        torch.manual_seed(42)
        with monkeypatch.context() as patch:
            patch.setattr(config, "CONDITIONED", enabled)
            model, _ = config.build_model(SimpleNamespace(vocab_size=300))
        models.append(model)
    a, b = models
    aa = a.state_dict()
    for name, value in b.state_dict().items():
        torch.testing.assert_close(aa[name], value, atol=0, rtol=0)
    x = torch.randint(300, (2, 13))
    torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
