from dataclasses import replace

import pytest
import torch

from src.model.cache_storage import CacheStorage, StateEntry
from src.model.composition import build_block
from src.model.definitions import CumulativeMixer
from src.model.stability import HeadRMSNormImpl, RMSNormImpl

MODES = [
    ("global", "elementwise"),
    ("head", "elementwise"),
    ("head", "matrix"),
    ("global", "matrix"),
    ("global", "head_mlp"),
]


def definition(scope, readout):
    return CumulativeMixer(
        8,
        normalization_scope=scope,
        readout=readout,
        num_heads=2,
        query_head_features=2,
        head_mlp_hidden=8,
        value_decay=True,
        decay_mode="token",
    )


def explicit_read(module, q, values):
    heads = values.reshape(*values.shape[:-1], module.num_heads, -1)
    queries = q.reshape(*q.shape[:-1], module.num_heads, module.query_head_features)
    outputs = []
    for h in range(module.num_heads):
        a, b = heads[..., h, :], heads[..., (h + 1) % module.num_heads, :]
        u = a @ module.pair_p[h]
        matrix = torch.diag_embed(u) @ module.pair_w[h] @ torch.diag_embed(b)
        outputs.append((queries[..., h, :].unsqueeze(-2) @ matrix).squeeze(-2))
    return torch.cat(outputs, -1)


def test_head_norm_reference_isolation_dtype_and_gains():
    norm = HeadRMSNormImpl(8, 1e-6, 2).double()
    norm.weights.data.copy_(torch.arange(1.0, 9.0))
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    h = x.reshape(2, 3, 2, 4)
    expected = (h / (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()).reshape_as(
        x
    ) * norm.weights
    torch.testing.assert_close(norm(x), expected)
    altered = x.clone()
    altered[..., 4:] *= 100
    torch.testing.assert_close(norm(x)[..., :4], norm(altered)[..., :4], rtol=0, atol=0)
    assert norm(x).dtype == x.dtype
    assert norm.weights.numel() == 8
    global_norm = RMSNormImpl(8, 1e-6).double()
    assert not torch.allclose(global_norm(x)[..., :4], global_norm(altered)[..., :4])


def test_factored_readout_explicit_cyclic_values_and_gradients():
    module = definition("head", "matrix").build().double()
    q = torch.rand(2, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    actual = module.read_values(q, v)
    expected = explicit_read(module, q, v)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    params = (q, v, module.pair_p, module.pair_w)
    weights = torch.randn_like(actual)
    left = torch.autograd.grad((actual * weights).sum(), params, retain_graph=True)
    right = torch.autograd.grad((expected * weights).sum(), params)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    assert torch.autograd.gradcheck(lambda a, b: module.read_values(a, b), (q, v))


@pytest.mark.parametrize("scope,readout", MODES)
def test_recurrence_reference_gradients_and_causality(scope, readout):
    module = definition(scope, readout).build().double()
    module.value_decay_gate.weight.data.normal_(0, 0.02)
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    q = module.q_proj(x).sigmoid()
    k = module.k_proj(x).tanh()
    v = module.v_proj(x).sigmoid()
    gate = (
        x @ module.value_decay_gate.weight.T + module.value_decay_gate.bias
    ).sigmoid()
    ks = torch.zeros_like(k[:, 0])
    vs = torch.zeros_like(v[:, 0])
    values = []
    for t in range(7):
        ks = ks + k[:, t]
        vs = gate[:, t] * vs + v[:, t] * module.key_norm(ks)
        values.append(module.value_norm(vs))
    expected = module.read_values(q, torch.stack(values, 1))
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    parameters = (
        x,
        module.k_proj.weight,
        module.v_proj.weight,
        module.value_decay_gate.weight,
    )
    left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), parameters, retain_graph=True)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    changed = x.detach().clone()
    changed[:, 4:] += 10
    torch.testing.assert_close(module(changed)[:, :4], actual[:, :4], rtol=0, atol=0)
    gradient = torch.autograd.grad(actual[:, :4].sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 4:]) == 0


@pytest.mark.parametrize("scope,readout", MODES)
def test_cached_chunks_single_tokens_snapshot_constant_state_and_residual(
    scope, readout, tmp_path
):
    spec = definition(scope, readout)
    module = spec.build().double()
    module.assign_id(1)
    x = torch.randn(2, 13, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        for sizes in ([3, 4, 6], [1] * 13):
            cache = CacheStorage(
                [spec.cache(1)], torch.device("cpu"), torch.device("cpu")
            )
            outputs = []
            start = 0
            for size in sizes:
                outputs.append(module(x[:, start : start + size], cache=cache))
                start += size
                cache.save(tmp_path / "cache.pt")
                cache = CacheStorage.load(
                    tmp_path / "cache.pt",
                    storage_device=torch.device("cpu"),
                    read_device=torch.device("cpu"),
                )
            torch.testing.assert_close(
                torch.cat(outputs, 1), expected, rtol=1e-10, atol=1e-10
            )
            state = cache.read_state(module.layer_id)
            assert (
                state.count == 13
                and state.key_sum.numel() == 16
                and state.value_sum.numel() == 16
            )
            assert isinstance(state, StateEntry) and not cache.entries
        torch.manual_seed(12)
        branch = spec.build().double()
        torch.manual_seed(12)
        residual = build_block(replace(spec, residual=True)).double()
        torch.testing.assert_close(residual(x), x + branch(x), rtol=0, atol=0)


def test_mlp_identity_cross_head_dependency_and_rng_preservation():
    torch.manual_seed(42)
    original = definition("global", "elementwise").build()
    torch.manual_seed(42)
    mixed = definition("global", "head_mlp").build()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, mixed.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 6, 8)
    torch.testing.assert_close(original(x), mixed(x), rtol=0, atol=0)
    # Deliberately route head 0 to head 1 through the shared MLP.
    with torch.no_grad():
        mixed.head_mlp[0].weight.zero_()
        mixed.head_mlp[0].bias.zero_()
        mixed.head_mlp[0].weight[0, 0] = 1
        mixed.head_mlp[2].weight.zero_()
        mixed.head_mlp[2].weight[1, 0] = 1
    v = torch.ones(1, 1, 8, requires_grad=True)
    output = mixed.read_values(torch.ones_like(v), v)
    gradient = torch.autograd.grad(output[..., 4:].sum(), v)[0]
    assert gradient[..., :4].abs().sum() > 0


def test_matrix_common_initialization_and_b_c_matching():
    torch.manual_seed(42)
    baseline = definition("global", "elementwise").build()
    torch.manual_seed(42)
    b = definition("head", "matrix").build()
    torch.manual_seed(42)
    c = definition("global", "matrix").build()
    for key, value in b.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[key], rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        if not key.startswith("q_proj"):
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("scope,readout", MODES)
def test_zero_long_sequence_and_double_gradcheck(scope, readout):
    module = definition(scope, readout).build().double()
    for x in (
        torch.zeros(1, 1025, 8, dtype=torch.float64),
        torch.randn(1, 1025, 8, dtype=torch.float64),
    ):
        output = module(x)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()
        )
        module.zero_grad()
    small = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (small,), fast_mode=True)


@pytest.mark.parametrize("scope,readout", MODES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_gradient_and_gate_precision(scope, readout):
    module = definition(scope, readout).build().cuda()
    x = torch.randn(2, 256, 8, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gate = module.value_decay_gate(x)
        torch.testing.assert_close(gate, torch.full_like(gate, 0.99), atol=1e-7, rtol=0)
        assert gate.dtype == torch.float32
        output = module(x)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()
    )
    assert module.k_proj.weight.grad.abs().sum() > 0
    assert module.value_decay_gate.weight.grad.abs().sum() > 0


def test_full_model_parameter_budget_and_initial_unchanged_weights():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, _ = build_model(Tokenizer())
    mode = next(m.readout for m in model.modules() if hasattr(m, "readout"))
    assert sum(p.numel() for p in model.parameters()) == (
        13775988 if mode == "head_mlp" else 13774080
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 3},
        {"num_heads": 0},
        {"readout": "bad"},
        {"normalization_scope": "bad"},
        {"readout": "matrix", "bias": True},
        {"readout": "head_mlp", "normalization_scope": "head"},
        {"query_head_features": 0},
        {"head_mlp_hidden": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CumulativeMixer(8, **kwargs)
