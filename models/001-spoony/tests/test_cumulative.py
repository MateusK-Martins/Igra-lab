import copy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import CumulativeMixer


def reference(module, x):
    q = module.q_proj(x).sigmoid()
    k = module.k_proj(x).tanh()
    v = module.v_proj(x).sigmoid()
    keys = torch.zeros_like(k[:, 0])
    values = torch.zeros_like(v[:, 0])
    offset = 1.0 if module.normalization == "soft_rms" else module.key_norm.eps
    key_decay = (
        module.key_decay_logits.sigmoid() if module.key_decay_logits is not None else 1
    )
    value_decay = (
        module.value_decay_logits.sigmoid()
        if module.value_decay_logits is not None
        else 1
    )
    if module.key_decay_gate is not None:
        key_decay = (
            x @ module.key_decay_gate.weight.T + module.key_decay_gate.bias
        ).sigmoid()
    if module.value_decay_gate is not None:
        value_decay = (
            x @ module.value_decay_gate.weight.T + module.value_decay_gate.bias
        ).sigmoid()
    output = []
    for token in range(x.shape[1]):
        keys = (
            key_decay[:, token]
            if isinstance(key_decay, torch.Tensor) and key_decay.ndim == 3
            else key_decay
        ) * keys + k[:, token]
        normalized_keys = (
            keys
            * torch.rsqrt(keys.square().mean(-1, keepdim=True) + offset)
            * getattr(module.key_norm, "weights", 1)
        )
        values = (
            value_decay[:, token]
            if isinstance(value_decay, torch.Tensor) and value_decay.ndim == 3
            else value_decay
        ) * values + v[:, token] * normalized_keys
        normalized_values = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + offset)
            * getattr(module.value_norm, "weights", 1)
        )
        output.append(q[:, token] * normalized_values)
    return torch.stack(output, dim=1)


@pytest.mark.parametrize("normalization", ["rms", "soft_rms"])
@pytest.mark.parametrize("decay_mode", ["none", "v", "kv"])
@pytest.mark.parametrize("decay_type", ["diagonal", "token"])
def test_output_and_gradients_match_independent_sequential_recurrence(
    normalization, decay_mode, decay_type
):
    torch.manual_seed(123)
    module = (
        CumulativeMixer(
            6,
            bias=True,
            normalization=normalization,
            key_decay=decay_mode == "kv",
            value_decay=decay_mode != "none",
            decay_mode=decay_type,
        )
        .build()
        .double()
    )
    with torch.no_grad():
        for norm in (module.key_norm, module.value_norm):
            if hasattr(norm, "weights"):
                norm.weights.uniform_(0.5, 1.5)
    with torch.no_grad():
        for gate in (module.key_decay_gate, module.value_decay_gate):
            if gate is not None:
                gate.weight.normal_(0, 0.2)
                gate.bias.uniform_(-1, 1)
    other = copy.deepcopy(module)
    x = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = module(x), reference(other, y)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for first, second in zip(module.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)


def test_residual_is_added_exactly_once_and_zero_input_is_finite():
    module = build_block(CumulativeMixer(4, residual=True), 7)
    x = torch.randn(2, 8, 4)
    torch.testing.assert_close(module(x), x + reference(module.branch, x))
    zeros = torch.zeros_like(x, requires_grad=True)
    output = module(zeros)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(zeros.grad).all()


@pytest.mark.parametrize("normalization", ["rms", "soft_rms"])
@pytest.mark.parametrize("decay_mode", ["none", "v", "kv"])
@pytest.mark.parametrize("decay_type", ["diagonal", "token"])
def test_future_values_and_gradients_cannot_affect_prefix(
    normalization, decay_mode, decay_type
):
    module = (
        CumulativeMixer(
            5,
            normalization=normalization,
            key_decay=decay_mode == "kv",
            value_decay=decay_mode != "none",
            decay_mode=decay_type,
        )
        .build()
        .double()
    )
    x = torch.randn(2, 11, 5, dtype=torch.float64, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 100
    torch.testing.assert_close(module(x)[:, :6], module(changed)[:, :6])
    module(x)[:, :6].sum().backward()
    assert torch.count_nonzero(x.grad[:, 6:]) == 0


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1, 1), (2, 4, 1), (7,)])
@pytest.mark.parametrize("normalization", ["rms", "soft_rms"])
@pytest.mark.parametrize("decay_mode", ["none", "v", "kv"])
@pytest.mark.parametrize("decay_type", ["diagonal", "token"])
def test_cached_chunks_and_snapshot_match_full_sequence(
    tmp_path, chunks, normalization, decay_mode, decay_type
):
    definition = CumulativeMixer(
        5,
        residual=True,
        normalization=normalization,
        key_decay=decay_mode == "kv",
        value_decay=decay_mode != "none",
        decay_mode=decay_type,
    )
    module = build_block(definition, 3).double()
    with torch.no_grad():
        for gate in (module.branch.key_decay_gate, module.branch.value_decay_gate):
            if gate is not None:
                gate.weight.normal_(0, 0.2)
                gate.bias.uniform_(-1, 1)
    storage = CacheStorage(
        [definition.cache(3)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, sum(chunks), 5, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        outputs = []
        offset = 0
        for size in chunks:
            outputs.append(module(x[:, offset : offset + size], cache=storage))
            offset += size
            storage.save(tmp_path / "state.pt")
            storage = CacheStorage.load(
                tmp_path / "state.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, atol=1e-12, rtol=1e-12
    )
    state = storage.read_state(module.branch.layer_id)
    assert state.key_sum.shape == (2, 1, 5)
    assert state.value_sum.shape == (2, 1, 5)
    assert state.count == sum(chunks)


def test_cache_does_not_grow_and_rejects_changed_batch_without_mutation():
    definition = CumulativeMixer(4)
    module = build_block(definition, 0)
    storage = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    with torch.no_grad():
        module(torch.randn(2, 300, 4), cache=storage)
        before = storage.read_state(module.layer_id)
        with pytest.raises(ValueError, match="batch"):
            module(torch.randn(1, 3, 4), cache=storage)
        after = storage.read_state(module.layer_id)
    assert after.count == 300
    assert after.key_sum.numel() == 8
    torch.testing.assert_close(after.key_sum, before.key_sum)
    torch.testing.assert_close(after.value_sum, before.value_sum)
    with pytest.raises(RuntimeError, match="no_grad"):
        module(torch.randn(2, 1, 4), cache=storage)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"input_features": True},
        {"input_features": 4, "eps": 0},
        {"input_features": 4, "eps": float("nan")},
        {"input_features": 4, "bias": 1},
        {"input_features": 4, "residual": 1},
        {"input_features": 4, "normalization": "bad"},
        {"input_features": 4, "decay_mode": "bad"},
        {"input_features": 4, "key_decay": 1},
        {"input_features": 4, "value_decay": 1},
        {"input_features": 4, "decay_init": 0},
        {"input_features": 4, "decay_init": 1},
        {"input_features": 4, "decay_init": float("nan")},
    ],
)
def test_invalid_definition(kwargs):
    with pytest.raises(ValueError):
        CumulativeMixer(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("normalization", ["rms", "soft_rms"])
@pytest.mark.parametrize("decay_mode", ["none", "v", "kv"])
@pytest.mark.parametrize("decay_type", ["diagonal", "token"])
def test_cuda_mixed_precision_gradients_and_cpu_cached_state(
    dtype, normalization, decay_mode, decay_type
):
    definition = CumulativeMixer(
        16,
        residual=True,
        normalization=normalization,
        key_decay=decay_mode == "kv",
        value_decay=decay_mode != "none",
        decay_mode=decay_type,
    )
    module = build_block(definition, 2).cuda()
    with torch.no_grad():
        for gate in (module.branch.key_decay_gate, module.branch.value_decay_gate):
            if gate is not None:
                gate.weight.normal_(0, 0.2)
    x = torch.randn(2, 33, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        full = module(x)
    full.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    storage = CacheStorage(
        [definition.cache(2)], torch.device("cpu"), torch.device("cuda")
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        chunks = torch.cat(
            [module(x[:, :11], cache=storage), module(x[:, 11:], cache=storage)], dim=1
        )
    torch.testing.assert_close(chunks, full, atol=0.03, rtol=0.03)
    state = storage.states[module.branch.layer_id]
    assert state.key_sum.device.type == "cpu"
    assert state.key_sum.dtype == torch.float32
