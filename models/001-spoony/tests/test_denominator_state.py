import copy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import DenominatorState, SignedState
from src.model.scan import affine_prefix_sum


def reference(m, x):
    q = m.q_proj(x).sigmoid()
    u = m.v_proj(x).tanh()
    g = (x @ m.value_decay_gate.weight.T + m.value_decay_gate.bias).sigmoid()
    s = torch.zeros_like(u[:, 0])
    z = torch.zeros_like(s)
    outs = []
    for t in range(x.shape[1]):
        s = g[:, t] * s + u[:, t]
        z = (g[:, t] if m.mode == "average" else g[:, t].square()) * z + 1
        divisor = z + m.eps
        if m.mode == "squared":
            divisor = divisor.sqrt()
        outs.append(q[:, t] * m.gain * s / divisor)
    return torch.stack(outs, 1), s, z


@pytest.mark.parametrize("mode", ["average", "squared"])
def test_independent_reference_values_and_gradients(mode):
    m = DenominatorState(4, mode=mode).build().double()
    with torch.no_grad():
        m.value_decay_gate.weight.normal_(0, 0.2)
        m.value_decay_gate.bias.uniform_(-1, 3)
        m.gain.uniform_(0.5, 1.5)
    other = copy.deepcopy(m)
    x = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a = m(x)
    b, _, _ = reference(other, y)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    w = torch.randn_like(a)
    (a * w).sum().backward()
    (b * w).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
    for (name, p), (_, r) in zip(
        m.named_parameters(), other.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-7, atol=1e-8)
    assert (
        m.value_decay_gate.weight.grad.abs().sum() > 0
        and m.v_proj.weight.grad.abs().sum() > 0
    )
    assert m.gain.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["average", "squared"])
def test_causality_gradcheck_and_residual_once(mode):
    d = DenominatorState(3, mode=mode)
    m = build_block(d, 0).double()
    x = torch.randn(1, 7, 3, dtype=torch.float64, requires_grad=True)
    out = m(x)
    changed = x.detach().clone()
    changed[:, 4:] += 3
    torch.testing.assert_close(out[:, :4], m(changed)[:, :4])
    grad = torch.autograd.grad(out[:, :4].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 4:]) == 0
    assert torch.autograd.gradcheck(
        m, (x[:, :2].detach().requires_grad_(),), fast_mode=True
    )
    d.residual = True
    wrapped = build_block(d, 0).double()
    wrapped.branch.load_state_dict(m.state_dict())
    torch.testing.assert_close(wrapped(x), x + m(x))


@pytest.mark.parametrize("mode", ["average", "squared"])
def test_cached_chunks_single_tokens_snapshot_and_constant_size(mode, tmp_path):
    d = DenominatorState(4, mode=mode)
    m = build_block(d, 0).double()
    x = torch.randn(2, 39, 4, dtype=torch.float64)
    cache = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        full = m(x)
        a = m(x[:, :9], cache=cache)
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        b = m(x[:, 9:21], cache=restored)
        c = torch.cat([m(x[:, t : t + 1], cache=restored) for t in range(21, 39)], 1)
        torch.testing.assert_close(torch.cat([a, b, c], 1), full, rtol=1e-8, atol=1e-9)
        _, s, z = reference(m, x)
        state = restored.read_state(m.layer_id)
        torch.testing.assert_close(state.value_sum[:, 0], s, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(state.key_sum[:, 0], z, rtol=1e-8, atol=1e-9)
        assert (state.key_sum >= 1).all()
        assert state.key_sum.numel() + state.value_sum.numel() == 2 * 2 * 4
        m(x, cache=restored)
        state = restored.read_state(m.layer_id)
        assert state.key_sum.numel() + state.value_sum.numel() == 2 * 2 * 4
        fresh = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
        torch.testing.assert_close(m(x, cache=fresh), full)
    with pytest.raises(RuntimeError, match="no_grad"):
        m(x, cache=cache)


def test_initialization_gain_numerator_identity_and_mode_cache_isolation():
    torch.manual_seed(42)
    a = build_block(DenominatorState(4, mode="average"), 0).double()
    end = torch.random.get_rng_state()
    torch.manual_seed(42)
    b = build_block(DenominatorState(4, mode="squared"), 0).double()
    assert torch.equal(end, torch.random.get_rng_state())
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
    assert torch.equal(a.gain, torch.ones_like(a.gain))
    torch.manual_seed(42)
    original = SignedState(4).build().double()
    for name, value in a.state_dict().items():
        torch.testing.assert_close(
            value,
            original.state_dict()["value_norm.weights" if name == "gain" else name],
            rtol=0,
            atol=0,
        )
    x = torch.randn(2, 27, 4, dtype=torch.float64)
    ca = CacheStorage(
        [DenominatorState(4, mode="average").cache(0)],
        torch.device("cpu"),
        torch.device("cpu"),
    )
    cb = CacheStorage(
        [DenominatorState(4, mode="squared").cache(0)],
        torch.device("cpu"),
        torch.device("cpu"),
    )
    with torch.no_grad():
        a(x, cache=ca)
        b(x, cache=cb)
        torch.testing.assert_close(
            ca.read_state(a.layer_id).value_sum,
            cb.read_state(b.layer_id).value_sum,
            rtol=0,
            atol=0,
        )
        assert (
            ca.read_state(a.layer_id).key_sum > cb.read_state(b.layer_id).key_sum
        ).all()
        with pytest.raises(KeyError):
            b(x, cache=ca)
    with pytest.raises(RuntimeError):
        a.load_state_dict(original.state_dict())


@pytest.mark.parametrize("g", [0.0, 0.5, 0.99, 1.0])
def test_constant_retention_denominator_closed_form(g):
    gate = torch.full((1, 4096, 1), g, dtype=torch.float64)
    ones = torch.ones_like(gate)
    for power in (1, 2):
        actual = affine_prefix_sum(ones, gate.pow(power))
        c = g**power
        t = torch.arange(1, 4097, dtype=torch.float64)
        expected = t if c == 1 else (1 - c**t) / (1 - c)
        torch.testing.assert_close(actual.flatten(), expected, rtol=1e-10, atol=1e-10)
    if g == 0.99:
        assert actual[0, -1, 0].item() == pytest.approx(1 / (1 - g * g), rel=1e-10)


def test_squared_weight_energy_explains_variance_scale():
    n = 64
    g = 0.9
    weights = g ** torch.arange(n - 1, -1, -1, dtype=torch.float64)
    p = affine_prefix_sum(
        torch.ones(1, n, 1, dtype=torch.float64),
        torch.full((1, n, 1), g * g, dtype=torch.float64),
    )[0, -1, 0]
    torch.testing.assert_close(weights.square().sum(), p)
    torch.testing.assert_close(
        (weights / p.sqrt()).square().sum(), torch.tensor(1.0, dtype=torch.float64)
    )


@pytest.mark.parametrize("mode", ["average", "squared"])
def test_zero_long_sequence_and_denominator_gradients(mode):
    m = DenominatorState(4, mode=mode).build()
    assert torch.equal(m(torch.zeros(1, 2049, 4)), torch.zeros(1, 2049, 4))
    out = m(torch.randn(1, 2049, 4))
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


@pytest.mark.parametrize("mode", ["average", "squared"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fp32_gate_states_and_gradient_stability(mode):
    d = DenominatorState(8, mode=mode)
    m = build_block(d, 0).cuda()
    x = torch.randn(2, 256, 8, device="cuda")
    cache = CacheStorage([d.cache(0)], torch.device("cuda"), torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).square().mean().backward()
        with torch.no_grad():
            m(x, cache=cache)
    state = cache.read_state(m.layer_id)
    assert state.key_sum.dtype == state.value_sum.dtype == torch.float32
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert m.value_decay_gate(x).mean().item() == pytest.approx(0.99, abs=1e-6)


def test_validation_and_actual_parameter_budget():
    from configs.model.model import build_model
    from src.model.stability import RMSNormImpl

    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    assert sum(p.numel() for p in model.parameters()) == 12887040
    assert sum(isinstance(m, RMSNormImpl) for m in model.modules()) == 12
    assert all(e.key_sum.shape[-1] == e.value_sum.shape[-1] == 384 for e in entries)
    for kw in [{"mode": "rms"}, {"eps": 0}, {"decay_init": 1}, {"input_features": 0}]:
        with pytest.raises(ValueError):
            DenominatorState(**({"input_features": 4} | kw))
