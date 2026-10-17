import copy
from dataclasses import replace

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import TideMixer


def reference(m, x):
    q = m.q_proj(x).sigmoid()
    writes = (
        m.v_proj(x).tanh() * (x @ m.write_gate.weight.T + m.write_gate.bias).sigmoid()
    )
    gate = (x @ m.value_decay_gate.weight.T + m.value_decay_gate.bias).sigmoid()
    state = torch.zeros_like(writes[:, 0])
    outputs = []
    for t in range(x.shape[1]):
        state = gate[:, t] * state + writes[:, t]
        read = (
            state
            * (state.square().mean(-1, keepdim=True) + m.value_norm.eps).rsqrt()
            * m.value_norm.weights
        )
        outputs.append(q[:, t] * read)
    return torch.stack(outputs, 1)


@pytest.fixture
def definition():
    return TideMixer(4)


def test_independent_reference_values_and_all_parameter_gradients(definition):
    m = definition.build().double()
    with torch.no_grad():
        m.value_decay_gate.weight.normal_(0, 0.2)
        if m.write_gate is not None:
            m.write_gate.weight.normal_(0, 0.2)
        m.value_norm.weights.uniform_(0.5, 1.5)
    other = copy.deepcopy(m)
    x = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a = m(x)
    b = reference(other, y)
    torch.testing.assert_close(a, b, rtol=1e-09, atol=1e-09)
    w = torch.randn_like(a)
    (a * w).sum().backward()
    (b * w).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-07, atol=1e-08)
    for (name, p), (_, r) in zip(
        m.named_parameters(), other.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-07, atol=1e-08)
    if m.write_gate is not None:
        assert m.write_gate.weight.grad.abs().sum() > 0


def test_causality_gradcheck_and_residual_once(definition):
    m = build_block(definition, 0).double()
    x = torch.randn(1, 7, 4, dtype=torch.float64, requires_grad=True)
    out = m(x)
    changed = x.detach().clone()
    changed[:, 4:] += 3
    torch.testing.assert_close(out[:, :4], m(changed)[:, :4])
    grad = torch.autograd.grad(out[:, :4].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 4:]) == 0
    assert torch.autograd.gradcheck(
        m, (x[:, :2].detach().requires_grad_(),), fast_mode=True
    )
    wrapped = build_block(replace(definition, residual=True), 0).double()
    wrapped.branch.load_state_dict(m.state_dict())
    torch.testing.assert_close(wrapped(x), x + m(x))


def test_cached_chunks_snapshot_single_token_reset_and_constant_size(
    definition, tmp_path
):
    m = build_block(definition, 0).double()
    x = torch.randn(2, 31, 4, dtype=torch.float64)
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    with torch.no_grad():
        full = m(x)
        a = m(x[:, :11], cache=cache)
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        b = torch.cat([m(x[:, t : t + 1], cache=restored) for t in range(11, 31)], 1)
        torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-08, atol=1e-09)
        state = restored.read_state(m.layer_id)
        assert state.key_sum.numel() == 0 and state.value_sum.numel() == 8
        m(x, cache=restored)
        assert restored.read_state(m.layer_id).value_sum.numel() == 8
        fresh = CacheStorage(
            [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
        )
        torch.testing.assert_close(m(x, cache=fresh), full)
    with pytest.raises(RuntimeError, match="no_grad"):
        m(x, cache=cache)


def test_write_gate_suppresses_writes_independently_from_retention():
    d = TideMixer(4)
    m = build_block(d, 0).double()
    cache = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    x = torch.randn(2, 9, 4, dtype=torch.float64)
    with torch.no_grad():
        m(x, cache=cache)
        old = cache.read_state(m.layer_id).value_sum.clone()
        m.write_gate.bias.fill_(-1000)
        m(x[:, :1], cache=cache)
        expected = old * m.value_decay_gate(x[:, :1])
        torch.testing.assert_close(cache.read_state(m.layer_id).value_sum, expected)
        assert m.value_decay_gate(x[:, :1])[..., -1].mean() > 0.99


def test_zero_long_sequence_finite_gradients(definition):
    m = definition.build()
    assert torch.equal(m(torch.zeros(1, 2049, 4)), torch.zeros(1, 2049, 4))
    out = m(torch.randn(1, 2049, 4))
    out.square().mean().backward()
    assert torch.isfinite(out).all() and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fp32_cache_and_finite_gradients(definition):
    m = build_block(definition, 0).cuda()
    x = torch.randn(2, 256, 4, device="cuda")
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cuda"), torch.device("cuda")
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).square().mean().backward()
        with torch.no_grad():
            m(x, cache=cache)
    assert cache.read_state(m.layer_id).value_sum.dtype == torch.float32
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    if m.write_gate is not None:
        torch.testing.assert_close(
            m.write_gate(x).mean((0, 1)),
            torch.linspace(0.20, 0.02, 4, device="cuda"),
            rtol=1e-6,
            atol=1e-7,
        )


def test_parameter_count_and_ranged_initialization():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    assert sum(p.numel() for p in model.parameters()) == 69_725_184
    assert len(entries) == 9
    assert all(
        e.key_sum.numel() == 0 and e.value_sum.shape == (0, 1, 768)
        for e in entries
        if hasattr(e, "key_sum")
    )
    for index in [1, 5, 13, 17, 25, 29]:
        m = model.body[index].branch
        torch.testing.assert_close(
            m.value_decay_gate.bias.sigmoid(), torch.linspace(0.90, 0.995, 768)
        )
        torch.testing.assert_close(
            m.write_gate.bias.sigmoid(), torch.linspace(0.20, 0.02, 768)
        )
        assert not hasattr(m, "k_proj")
        assert not hasattr(m, "read_up")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"eps": 0},
        {"write_initials": (0.1,)},
        {"retention_initials": (0.9, 0.95, 0.99, 1.0)},
        {"write_initials": (0.2, 0.1, float("nan"), 0.02)},
    ],
)
def test_invalid_definitions(kwargs):
    with pytest.raises(ValueError):
        TideMixer(**({"input_features": 4} | kwargs))
