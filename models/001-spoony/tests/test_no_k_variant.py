import copy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import NoKVariant, SignedState


def reference(m, x):
    logits = m.q_proj(x)
    q = logits.tanh() if m.mode == "signed_readout" else logits.sigmoid()
    u = m.v_proj(x).tanh()
    if m.write_gate is not None:
        u = u * (x @ m.write_gate.weight.T + m.write_gate.bias).sigmoid()
    gate = (x @ m.value_decay_gate.weight.T + m.value_decay_gate.bias).sigmoid()
    state = torch.zeros_like(u[:, 0])
    outputs = []
    for t in range(x.shape[1]):
        state = gate[:, t] * state + u[:, t]
        read = (
            state
            * (
                state.reshape(
                    *state.shape[:-1], getattr(m.value_norm, "num_heads", 1), -1
                )
                .square()
                .mean(-1, keepdim=True)
                .expand(
                    *state.shape[:-1],
                    getattr(m.value_norm, "num_heads", 1),
                    state.shape[-1] // getattr(m.value_norm, "num_heads", 1),
                )
                .flatten(-2)
                + m.value_norm.eps
            ).rsqrt()
            * m.value_norm.weights
        )
        if m.read_down is not None:
            matrix = m.read_up.weight @ m.read_down.weight
            read = read @ matrix.T
        outputs.append(q[:, t] * read)
    return torch.stack(outputs, 1)


@pytest.fixture(params=["write_gate", "signed_readout", "lowrank"])
def definition(request):
    return NoKVariant(4, mode=request.param, rank=2)


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
    if m.write_gate is not None:
        assert m.write_gate.weight.grad.abs().sum() > 0
    if m.read_down is not None:
        assert (
            m.read_down.weight.grad.abs().sum() > 0
            and m.read_up.weight.grad.abs().sum() > 0
        )


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
    definition.residual = True
    wrapped = build_block(definition, 0).double()
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
        torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-8, atol=1e-9)
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


def test_shared_parameter_initialization_matches_NoK(definition):
    torch.manual_seed(42)
    m = definition.build()
    end = torch.random.get_rng_state()
    torch.manual_seed(42)
    original = SignedState(4).build()
    assert torch.equal(end, torch.random.get_rng_state())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, m.state_dict()[name], rtol=0, atol=0)
    if m.write_gate is not None:
        torch.testing.assert_close(
            m.write_gate(torch.randn(2, 7, 4)),
            torch.full((2, 7, 4), 0.5),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(m.write_gate.weight) == 0


def test_write_gate_suppresses_writes_independently_from_retention():
    d = NoKVariant(4, mode="write_gate", rank=2)
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
        assert m.value_decay_gate(x[:, :1]).mean() > 0.98


def test_signed_query_can_reverse_the_same_stored_feature():
    m = NoKVariant(2, mode="signed_readout", rank=1).build().double()
    with torch.no_grad():
        m.q_proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        m.v_proj.weight.copy_(
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        )
    x = torch.tensor([[[-1.0, 1.0]]], dtype=torch.float64)
    state = m.v_proj(x).tanh()
    assert (state > 0).all()
    result = m(x)
    assert result[0, 0, 0] < 0 and result[0, 0, 1] > 0


def test_lowrank_direct_matrix_factoring_and_gradients():
    m = NoKVariant(4, mode="lowrank", rank=2).build().double()
    z = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    factored = m.read_up(m.read_down(z))
    explicit = z @ (m.read_up.weight @ m.read_down.weight).T
    torch.testing.assert_close(factored, explicit)
    a = torch.autograd.grad(
        factored.square().sum(),
        (z, m.read_down.weight, m.read_up.weight),
        retain_graph=True,
    )
    b = torch.autograd.grad(
        explicit.square().sum(), (z, m.read_down.weight, m.read_up.weight)
    )
    for p, q in zip(a, b, strict=True):
        torch.testing.assert_close(p, q)
    assert torch.linalg.matrix_rank(m.read_up.weight @ m.read_down.weight) <= 2


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
        assert m.write_gate(x).mean().item() == pytest.approx(0.5, abs=1e-6)


def test_configuration_parameter_counts_and_state_budget():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    m = model.body[1].branch
    assert sum(p.numel() for p in model.parameters()) == 12_652_160
    states = [e for e in entries if hasattr(e, "value_sum")]
    assert len(states) == 4
    assert sorted(e.value_sum.shape[-1] for e in states) == [384, 384, 32768, 32768]
    assert all(e.key_sum.numel() == 0 for e in states)
    if m.mode == "lowrank":
        assert m.rank == 32
    for kw in [
        {"mode": "other"},
        {"rank": 0},
        {"mode": "lowrank", "rank": 5},
        {"eps": 0},
    ]:
        with pytest.raises(ValueError):
            NoKVariant(**({"input_features": 4} | kw))
