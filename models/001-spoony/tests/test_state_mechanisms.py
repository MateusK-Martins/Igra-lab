import copy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import MassNormalized, SignedState


def reference(m, x):
    q = m.q_proj(x).sigmoid()
    v = m.v_proj(x)
    mass = hasattr(m, "eps") and not hasattr(m, "value_norm")
    k = m.k_proj(x).tanh() if mass else None
    if mass:
        v = v.sigmoid()
    else:
        v = v.tanh()
    gate = (
        (x @ m.value_decay_gate.weight.T + m.value_decay_gate.bias).sigmoid()
        if m.value_decay_gate is not None
        else m.value_decay_logits.sigmoid()
    )
    state = torch.zeros_like(v[:, 0])
    kn = torch.zeros_like(state)
    kd = torch.zeros_like(state)
    vd = torch.zeros_like(state)
    out = []
    for t in range(x.shape[1]):
        g = gate[:, t] if gate.ndim == 3 else gate
        if mass:
            kn = kn + k[:, t]
            kd = kd + k[:, t].abs()
            used = kn / kd.clamp_min(m.eps)
            state = g * state + v[:, t] * used
            vd = g * vd + used.abs()
            read = state / vd.clamp_min(m.eps)
        else:
            state = g * state + v[:, t]
            read = (
                (
                    state
                    * (state.square().mean(-1, keepdim=True) + m.value_norm.eps).rsqrt()
                    * m.value_norm.weights
                )
                if m.normalize_state
                else state
            )
        out.append(q[:, t] * read)
    return torch.stack(out, 1)


@pytest.fixture(
    params=["token", "diagonal", "mass", "token-no-norm", "diagonal-no-norm"]
)
def definition(request):
    if request.param == "mass":
        return MassNormalized(4)
    return SignedState(
        4,
        token_decay=request.param.startswith("token"),
        normalize_state=not request.param.endswith("no-norm"),
    )


def test_independent_reference_and_all_parameter_gradients(definition):
    m = definition.build().double()
    other = copy.deepcopy(m)
    with torch.no_grad():
        if m.value_decay_gate is not None:
            m.value_decay_gate.weight.normal_(0, 0.2)
            other.load_state_dict(m.state_dict())
    x = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a = m(x)
    b = reference(other, y)
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)
    w = torch.randn_like(a)
    (a * w).sum().backward()
    (b * w).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
    for (name, p), (_, r) in zip(
        m.named_parameters(), other.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-7, atol=1e-8)
    decay = (
        m.value_decay_gate.weight
        if m.value_decay_gate is not None
        else m.value_decay_logits
    )
    assert decay.grad.abs().sum() > 0


def test_causality_gradcheck_and_residual_once(definition):
    m = build_block(definition, 0).double()
    x = torch.randn(1, 7, 4, dtype=torch.float64, requires_grad=True)
    out = m(x)
    future = x.detach().clone()
    future[:, 4:] += 2
    torch.testing.assert_close(out[:, :4], m(future)[:, :4])
    grad = torch.autograd.grad(out[:, :4].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 4:]) == 0
    assert torch.autograd.gradcheck(
        m, (x[:, :2].detach().requires_grad_(),), fast_mode=True
    )
    definition.residual = True
    wrapped = build_block(definition, 0).double()
    wrapped.branch.load_state_dict(m.state_dict())
    torch.testing.assert_close(wrapped(x), x + m(x))


def test_chunks_snapshot_constant_memory_reset(definition, tmp_path):
    m = build_block(definition, 0).double()
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 31, 4, dtype=torch.float64)
    with torch.no_grad():
        full = m(x)
        a = m(x[:, :11], cache=cache)
        path = tmp_path / "snapshot.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        b = torch.cat([m(x[:, t : t + 1], cache=restored) for t in range(11, 31)], 1)
        torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-8, atol=1e-9)
        state = restored.read_state(m.layer_id)
        size = state.key_sum.numel() + state.value_sum.numel()
        m(x, cache=restored)
        new = restored.read_state(m.layer_id)
        assert new.key_sum.numel() + new.value_sum.numel() == size
        if hasattr(m, "token_decay"):
            assert state.key_sum.numel() == 0
        else:
            _, kd = state.key_sum.chunk(2, -1)
            vn, vd = state.value_sum.chunk(2, -1)
            assert (kd >= 0).all() and (vd >= 0).all()
            assert (vn.abs() <= vd + 1e-10).all()
        fresh = CacheStorage(
            [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
        )
        torch.testing.assert_close(m(x, cache=fresh), full)
    with pytest.raises(RuntimeError, match="no_grad"):
        m(x, cache=cache)


def test_zero_and_long_sequence_stability(definition):
    m = definition.build()
    zero = m(torch.zeros(1, 2049, 4))
    assert torch.equal(zero, torch.zeros_like(zero))
    out = m(torch.randn(1, 2049, 4))
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    if isinstance(definition, MassNormalized):
        assert not hasattr(m, "key_norm")
        assert out.abs().max() <= 1.00001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fp32_state_and_gradients(definition):
    m = build_block(definition, 0).cuda()
    x = torch.randn(2, 256, 4, device="cuda")
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cuda"), torch.device("cuda")
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).square().mean().backward()
        with torch.no_grad():
            m(x, cache=cache)
    state = cache.read_state(m.layer_id)
    assert state.key_sum.dtype == state.value_sum.dtype == torch.float32
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_actual_parameter_count_and_no_k_storage():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    m = model.body[1].branch
    count = sum(p.numel() for p in model.parameters())
    if hasattr(m, "num_keys"):
        assert m.num_keys == 2 and not m.tied_query
        expected = 15552768
    elif hasattr(m, "token_decay"):
        expected = (12887040 if m.token_decay else 12002304) - (
            0 if m.normalize_state else 2304
        )
        assert not hasattr(m, "k_proj") and not hasattr(m, "key_norm")
        assert all(e.key_sum.numel() == 0 for e in entries)
    else:
        expected = 13769472
        assert not hasattr(m, "key_norm") and not hasattr(m, "value_norm")
    assert count == expected


def test_only_internal_value_norm_is_removed_and_initialization_preserved():
    from src.model.stability import RMSNormImpl

    torch.manual_seed(42)
    normalized = SignedState(4).build()
    after = torch.random.get_rng_state()
    torch.manual_seed(42)
    raw = SignedState(4, normalize_state=False).build()
    assert torch.equal(after, torch.random.get_rng_state())
    assert not hasattr(raw, "value_norm")
    assert not hasattr(raw, "key_norm")
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, normalized.state_dict()[name], rtol=0, atol=0)
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, _ = build_model(Tokenizer())
    assert sum(isinstance(m, RMSNormImpl) for m in model.modules()) == 12
    with pytest.raises(ValueError, match="normalize_state"):
        SignedState(4, normalize_state=1)
