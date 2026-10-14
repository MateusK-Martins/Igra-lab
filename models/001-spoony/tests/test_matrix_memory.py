import copy
from dataclasses import replace

import pytest
import torch

from src.model.cache_storage import CacheStorage, MatrixStateEntry, StateEntry
from src.model.composition import build_block
from src.model.definitions import CumulativeMixer, MatrixMemory
from src.model.matrix_memory import matrix_reference


def spec(**kwargs):
    return MatrixMemory(8, key_features=3, value_features=4, chunk_size=3, **kwargs)


def reference(module, x):
    q = module.q_down(module.q_proj(x).sigmoid())
    k = module.k_down(module.k_proj(x).tanh()) / module.key_features**0.5
    v = module.v_down(module.v_proj(x).sigmoid())
    a = (x @ module.row_gate.weight.T + module.row_gate.bias).sigmoid()
    b = (x @ module.column_gate.weight.T + module.column_gate.bias).sigmoid()
    state = x.new_zeros(x.shape[0], module.key_features, module.value_features)
    reads = []
    for t in range(x.shape[1]):
        state = (
            a[:, t, :, None] * b[:, t, None, :] * state
            + k[:, t, :, None] * v[:, t, None, :]
        )
        reads.append((q[:, t, None, :] @ state).squeeze(1) / module.key_features**0.5)
    return module.out_proj(module.read_norm(torch.stack(reads, 1))), state


def test_rank_one_writes_and_historical_retention():
    k = torch.tensor(
        [[[1.0, 2.0], [2.0, 1.0]]], dtype=torch.float64, requires_grad=True
    )
    v = torch.tensor(
        [[[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]]], dtype=torch.float64, requires_grad=True
    )
    gate = torch.full((1, 2, 2, 3), 0.5, dtype=torch.float64, requires_grad=True)
    states = matrix_reference(k, v, gate)
    torch.testing.assert_close(
        states[:, 0],
        torch.tensor([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        states[:, 1], 0.5 * states[:, 0] + k[:, 1, :, None] * v[:, 1, None, :]
    )
    dk, dv, dg = torch.autograd.grad(states[:, 1].sum(), (k, v, gate))
    assert (
        dk[:, 0].abs().sum() > 0
        and dv[:, 0].abs().sum() > 0
        and dg[:, 1].abs().sum() > 0
    )


def test_scan_reference_all_parameter_gradients_and_checkpointing():
    module = spec().build().double()
    module.row_gate.weight.data.normal_(0, 0.03)
    module.column_gate.weight.data.normal_(0, 0.03)
    x = torch.randn(2, 11, 8, dtype=torch.float64, requires_grad=True)
    actual = module(x)
    expected, last = reference(module, x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    parameters = (x, *module.parameters())
    left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8)
    plain = copy.deepcopy(module)
    plain.checkpoint_chunks = False
    x2 = x.detach().clone().requires_grad_()
    torch.testing.assert_close(plain(x2), actual, rtol=0, atol=0)
    small = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (small,), fast_mode=True)
    assert last.shape == (2, 3, 4)


def test_gate_initialization_and_learned_input_dependence():
    module = spec().build().double()
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    *_, a, b = module.projections(x)
    torch.testing.assert_close(
        a[..., :, None] * b[..., None, :],
        torch.full((2, 9, 3, 4), 0.99, dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )
    assert torch.count_nonzero(module.row_gate.weight) == 0
    assert torch.count_nonzero(module.column_gate.weight) == 0
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    module(x).square().mean().backward()
    assert module.row_gate.weight.grad.abs().sum() > 0
    assert module.column_gate.weight.grad.abs().sum() > 0
    optimizer.step()
    *_, a, b = module.projections(x)
    assert (a[:, 1:] - a[:, :1]).abs().max() > 0
    assert (b[:, 1:] - b[:, :1]).abs().max() > 0


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64])
def test_cached_full_chunked_single_token_equivalence_and_persistence(
    chunk_size, tmp_path
):
    definition = replace(spec(), chunk_size=chunk_size)
    module = definition.build().double()
    module.assign_id(1)
    module.row_gate.weight.data.normal_(0, 0.02)
    x = torch.randn(2, 17, 8, dtype=torch.float64)
    with torch.no_grad():
        full = module(x)
        expected, last = reference(module, x)
        torch.testing.assert_close(full, expected, rtol=1e-10, atol=1e-10)
        for sizes in ([5, 3, 9], [1] * 17):
            cache = CacheStorage(
                [definition.cache(1)], torch.device("cpu"), torch.device("cpu")
            )
            outputs = []
            start = 0
            for length in sizes:
                outputs.append(module(x[:, start : start + length], cache=cache))
                start += length
                cache.save(tmp_path / "state.pt")
                cache = CacheStorage.load(
                    tmp_path / "state.pt",
                    storage_device=torch.device("cpu"),
                    read_device=torch.device("cpu"),
                )
            torch.testing.assert_close(
                torch.cat(outputs, 1), full, rtol=1e-9, atol=1e-9
            )
            state = cache.read_matrix_state(module.layer_id)
            torch.testing.assert_close(state.matrix, last, rtol=1e-10, atol=1e-10)
            assert state.count == 17 and state.matrix.numel() == 24
            assert (
                cache.step(module.layer_id) == 17
                and not cache.entries
                and not cache.states
            )
        fresh = CacheStorage(
            [definition.cache(1)], torch.device("cpu"), torch.device("cpu")
        )
        torch.testing.assert_close(module(x, cache=fresh), full, rtol=0, atol=0)


def test_causality_and_future_gradient_isolation():
    module = spec().build().double()
    x = torch.randn(2, 13, 8, dtype=torch.float64, requires_grad=True)
    y = module(x)
    changed = x.detach().clone()
    changed[:, 5:] += 10
    torch.testing.assert_close(module(changed)[:, :5], y[:, :5], rtol=0, atol=0)
    gradient = torch.autograd.grad(y[:, :5].sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 5:]) == 0


def test_query_readout_and_residual_once():
    module = spec().build().double()
    q = torch.randn(1, 5, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 5, 4, dtype=torch.float64)
    a = torch.full_like(k, 0.9)
    b = torch.full_like(v, 0.8)
    initial = torch.randn(1, 3, 4, dtype=torch.float64)
    read, last = module._chunk(q, k, v, a, b, initial)
    states = matrix_reference(k, v, a[:, :, :, None] * b[:, :, None, :], initial)
    expected = torch.einsum("btk,btkv->btv", q, states) / 3**0.5
    torch.testing.assert_close(read, expected, rtol=1e-11, atol=1e-11)
    other, _ = module._chunk(q.flip(-1), k, v, a, b, initial)
    assert not torch.allclose(read, other)
    torch.testing.assert_close(last, states[:, -1])
    torch.manual_seed(42)
    plain = spec().build()
    torch.manual_seed(42)
    wrapped = build_block(replace(spec(), residual=True))
    x = torch.randn(1, 5, 8)
    torch.testing.assert_close(wrapped(x), x + plain(x), rtol=0, atol=0)


def test_long_sequence_zero_input_and_finite_gradients():
    module = replace(spec(), chunk_size=32).build()
    for x in (torch.zeros(1, 2049, 8), torch.randn(1, 2049, 8)):
        output = module(x)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        )
        module.zero_grad()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_fp32_state_and_gradients():
    definition = spec()
    module = definition.build().cuda()
    module.assign_id(1)
    x = torch.randn(2, 256, 8, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        *_, a, b = module.projections(x)
        assert a.dtype == b.dtype == torch.float32
        torch.testing.assert_close(
            a[..., :, None] * b[..., None, :],
            torch.full((2, 256, 3, 4), 0.99, device="cuda"),
            rtol=0,
            atol=2e-7,
        )
        output = module(x)
        loss = output.float().square().mean()
    loss.backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    for name in ("k_down", "v_down", "q_down", "row_gate", "column_gate"):
        assert getattr(module, name).weight.grad.abs().sum() > 0
    cache = CacheStorage(
        [definition.cache(1)], torch.device("cuda"), torch.device("cuda")
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module(x, cache=cache)
    assert cache.read_matrix_state(module.layer_id).matrix.dtype == torch.float32


def test_reject_legacy_vector_cache_and_weights(tmp_path):
    module = spec().build()
    module.assign_id(1)
    legacy = CacheStorage(
        [StateEntry(module.layer_id, torch.empty(0, 1, 8), torch.empty(0, 1, 8))],
        torch.device("cpu"),
        torch.device("cpu"),
    )
    with torch.no_grad(), pytest.raises(ValueError, match="MatrixStateEntry"):
        module(torch.randn(1, 2, 8), cache=legacy)
    with pytest.raises(RuntimeError):
        module.load_state_dict(CumulativeMixer(8).build().state_dict())
    definition = spec()
    cache = CacheStorage(
        [definition.cache(1)], torch.device("cpu"), torch.device("cpu")
    )
    cache.save(tmp_path / "state.pt")
    saved = torch.load(tmp_path / "state.pt", weights_only=True)
    for value in (
        torch.zeros(2, 3),
        torch.zeros(1, 3, 4, dtype=torch.bfloat16),
        torch.full((1, 3, 4), float("nan")),
    ):
        bad = copy.deepcopy(saved)
        bad["matrix_states"][0]["matrix"] = value
        torch.save(bad, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            CacheStorage.load(
                tmp_path / "bad.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )


def test_parameter_count_and_matrix_storage():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    assert sum(p.numel() for p in model.parameters()) == 14211456
    assert abs(14211456 / 13774080 - 1) < 0.05
    assert len(entries) == 6 and all(isinstance(e, MatrixStateEntry) for e in entries)
    assert all(e.matrix.shape == (0, 64, 128) for e in entries)
    assert 64 * 128 == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key_features": 0},
        {"value_features": 0},
        {"chunk_size": 0},
        {"decay_init": 1.0},
        {"eps": 0.0},
        {"checkpoint_chunks": 1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MatrixMemory(8, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_training_configuration_peak_memory_and_finite_backward():
    from configs.model.model import build_model

    class Tokenizer:
        vocab_size = 8192

    torch.cuda.empty_cache()
    model, _ = build_model(Tokenizer())
    model.cuda().train()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(8192, (8, 256), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), x.flatten())
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    peak = torch.cuda.max_memory_allocated()
    assert peak < torch.cuda.get_device_properties(0).total_memory * 0.9
