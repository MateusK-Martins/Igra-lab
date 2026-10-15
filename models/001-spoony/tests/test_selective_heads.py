import copy

import pytest
import torch

from src.model.definitions import NoKVariant
from src.model.stability import HeadRMSNormImpl
from tests.test_no_k_variant import (
    test_bf16_fp32_cache_and_finite_gradients as check_bf16,
)
from tests.test_no_k_variant import (
    test_cached_chunks_snapshot_single_token_reset_and_constant_size as check_cache,
)
from tests.test_no_k_variant import (
    test_causality_gradcheck_and_residual_once as check_causality,
)
from tests.test_no_k_variant import (
    test_independent_reference_values_and_all_parameter_gradients as check_reference,
)
from tests.test_no_k_variant import (
    test_zero_long_sequence_finite_gradients as check_long,
)


@pytest.fixture(params=["uniform", "specialized", "global", "retention", "write"])
def heads(request):
    return NoKVariant(
        4,
        num_heads=4,
        normalization_scope="global" if request.param == "global" else "head",
        retention_initials=(0.90, 0.95, 0.98, 0.995)
        if request.param in ("specialized", "global", "retention")
        else None,
        write_initials=(0.20, 0.10, 0.05, 0.02)
        if request.param in ("specialized", "global", "write")
        else None,
    )


def test_sequential_reference_values_and_gradients(heads):
    check_reference(heads)


def test_causality_gradcheck_residual(heads):
    check_causality(heads)


def test_cache_snapshots_chunks_constant_memory(heads, tmp_path):
    check_cache(heads, tmp_path)


def test_long_and_zero_finite(heads):
    check_long(heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16(heads):
    # Helper checks a uniform initial write gate; test heterogeneous precision below.
    uniform = copy.deepcopy(heads)
    uniform.write_initials = None
    check_bf16(uniform)
    m = heads.build().cuda()
    x = torch.randn(2, 256, heads.input_features, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x).square().mean().backward()
        actual = m.value_decay_gate(x).reshape(2, 256, 4, -1)
    expected = torch.tensor(heads.retention_initials or (0.99,) * 4, device="cuda")
    torch.testing.assert_close(actual.mean((0, 1, 3)), expected, rtol=1e-6, atol=1e-7)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_initial_preferences_and_common_parameters(heads):
    torch.manual_seed(42)
    m = heads.build()
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    original = NoKVariant(heads.input_features).build()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in original.state_dict().items():
        if name not in ("value_decay_gate.bias", "write_gate.bias"):
            torch.testing.assert_close(value, m.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2, 9, heads.input_features)
    for module, values in [
        (m.value_decay_gate, heads.retention_initials or (0.99,) * 4),
        (m.write_gate, heads.write_initials or (0.5,) * 4),
    ]:
        assert torch.count_nonzero(module.weight) == 0
        torch.testing.assert_close(
            module(x)[0, 0],
            torch.tensor(values).repeat_interleave(heads.input_features // 4),
        )
        with torch.no_grad():
            module.weight.normal_(0, 0.3)
        assert module(x).std(1).mean() > 0
    assert not hasattr(m, "k_proj")
    assert sum(p.numel() for p in m.parameters()) == sum(
        p.numel() for p in original.parameters()
    )


def test_headnorm_reference_and_group_isolation():
    m = HeadRMSNormImpl(8, 1e-6, 4).double()
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    grouped = x.reshape(2, 7, 4, 2)
    ref = (grouped / (grouped.square().mean(-1, keepdim=True) + 1e-6).sqrt()).flatten(
        -2
    )
    torch.testing.assert_close(m(x), ref)
    changed = x.detach().clone()
    changed[..., 2:] *= 100
    torch.testing.assert_close(m(x)[..., :2], m(changed)[..., :2])
    grad = torch.autograd.grad(m(x)[..., :2].sum(), x)[0]
    assert torch.count_nonzero(grad[..., 2:]) == 0


def test_invalid_heads_preferences():
    for kw in (
        {"num_heads": 3},
        {"num_heads": 0},
        {"num_heads": 4, "write_initials": (0.5,)},
        {"num_heads": 4, "retention_initials": (0.9, 0.95, 1, 0.99)},
        {"num_heads": 4, "mode": "lowrank"},
        {"write_initials": (float("nan"),)},
    ):
        with pytest.raises((TypeError, ValueError)):
            NoKVariant(8, **kw)


@pytest.mark.parametrize("scope", ["head", "global"])
def test_multifeature_heads_reference_and_partitioned_projection_rows(scope):
    from tests.test_no_k_variant import reference

    d = NoKVariant(
        12,
        num_heads=4,
        normalization_scope=scope,
        retention_initials=(0.9, 0.95, 0.98, 0.995),
        write_initials=(0.2, 0.1, 0.05, 0.02),
    )
    m = d.build().double()
    with torch.no_grad():
        m.write_gate.weight.normal_(0, 0.2)
        m.value_decay_gate.weight.normal_(0, 0.2)
    other = copy.deepcopy(m)
    x = torch.randn(2, 17, 12, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    out, ref = m(x), reference(other, y)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    (ref * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-7, atol=1e-8)
    # Each group's projections see all input features, not just its state slice.
    for gate in (m.q_proj, m.v_proj, m.write_gate, m.value_decay_gate):
        assert gate.weight.shape == (12, 12)


def test_global_norm_matches_original_and_preserves_same_recurrent_state():
    from src.model.cache_storage import CacheStorage
    from src.model.composition import build_block
    from src.model.stability import RMSNormImpl

    args = {
        "input_features": 12,
        "num_heads": 4,
        "retention_initials": (0.9, 0.95, 0.98, 0.995),
        "write_initials": (0.2, 0.1, 0.05, 0.02),
    }
    h = NoKVariant(**args)
    g = NoKVariant(**args, normalization_scope="global")
    torch.manual_seed(42)
    head = build_block(h, 0).double()
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    global_ = build_block(g, 0).double()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in head.state_dict().items():
        torch.testing.assert_close(value, global_.state_dict()[name], rtol=0, atol=0)
    norm = RMSNormImpl(12, 1e-6).double()
    x = torch.randn(2, 31, 12, dtype=torch.float64)
    assert torch.equal(norm(x), global_.value_norm(x))
    caches = [
        CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
        for d in (h, g)
    ]
    with torch.no_grad():
        head(x, cache=caches[0])
        global_(x, cache=caches[1])
    torch.testing.assert_close(
        caches[0].read_state(head.layer_id).value_sum,
        caches[1].read_state(global_.layer_id).value_sum,
        rtol=0,
        atol=0,
    )
    assert head.layer_id != global_.layer_id
    with pytest.raises(ValueError, match="normalization_scope"):
        NoKVariant(12, normalization_scope="invalid")
