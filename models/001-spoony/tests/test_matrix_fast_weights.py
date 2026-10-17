from types import SimpleNamespace

import pytest
import torch

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import MatrixFastWeights


def sequential(module, x, initial=None):
    decay = module._decays(device=x.device).to(x.dtype)
    raw = module.matrix_proj(x)
    state = (
        x.new_zeros(x.shape[0], module.scale_count, 64)
        if initial is None
        else initial[:, 0].to(x.dtype)
    )
    states = []
    for token in range(x.shape[1]):
        state = decay * state + (1 - decay) * raw[:, token, None, :]
        states.append(state)
    return torch.stack(states, dim=1), raw


@pytest.mark.parametrize("scales", [(0.99,), (0.99, 0.90, 0.99, 0.999)])
def test_recurrence_matches_sequential_reference_and_gradients(scales):
    torch.manual_seed(7)
    definition = MatrixFastWeights(12, timescales=scales)
    module = definition.build().double()
    x = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    actual = module._scan(
        module.matrix_proj(x)
        .to(torch.float64)
        .unsqueeze(2)
        .expand(-1, -1, len(scales), -1),
        module._decays(device=x.device),
        x.new_zeros(2, 1, len(scales), 64),
    )
    expected, _ = sequential(module, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    variables = (x, *module.parameters())
    ga = torch.autograd.grad(
        actual.square().sum(), variables, retain_graph=True, allow_unused=True
    )
    gb = torch.autograd.grad(expected.square().sum(), variables, allow_unused=True)
    for left, right in zip(ga, gb, strict=True):
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("scales", [(0.99,), (0.99, 0.90, 0.99, 0.999)])
def test_query_matches_matrix_multiplication_and_shape(scales):
    module = MatrixFastWeights(12, timescales=scales).build().double()
    x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    states, _ = sequential(module, x)
    q = module.query_proj(x)
    if len(scales) == 1:
        selected = states[:, :, 0]
    else:
        weights = module.scale_proj(x).softmax(-1)
        selected = torch.einsum("btn,btnf->btf", weights, states)
    expected = module.out_proj(
        torch.einsum("btij,btj->bti", selected.reshape(2, 5, 8, 8), q)
    )
    actual = module(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual.shape == x.shape
    assert torch.isfinite(torch.autograd.grad(actual.square().sum(), x)[0]).all()


@pytest.mark.parametrize("scales", [(0.99,), (0.99, 0.90, 0.99, 0.999)])
@pytest.mark.parametrize("chunk", [1, 3, 9])
def test_cached_chunks_match_full_and_restore_constant_state(scales, chunk, tmp_path):
    definition = MatrixFastWeights(12, timescales=scales, residual=True)
    module = build_block(definition, 2).double().eval()
    cache = CacheStorage(
        [definition.cache(2)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 11, 12, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat([module(part, cache=cache) for part in x.split(chunk, 1)], 1)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
        state = cache.states[module.branch.layer_id].state.clone()
        assert state.numel() == 2 * len(scales) * 64
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        next_x = torch.randn(2, 1, 12, dtype=torch.float64)
        torch.testing.assert_close(
            module(next_x, cache=cache), module(next_x, cache=restored)
        )
        torch.testing.assert_close(
            restored.states[module.branch.layer_id].state,
            cache.states[module.branch.layer_id].state,
        )
    assert cache.states[module.branch.layer_id].state.numel() == 2 * len(scales) * 64


def test_future_tokens_do_not_change_earlier_output_or_gradients():
    module = (
        MatrixFastWeights(12, timescales=(0.99, 0.90, 0.99, 0.999)).build().double()
    )
    x = torch.randn(2, 8, 12, dtype=torch.float64, requires_grad=True)
    y = module(x)
    changed = x.detach().clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(module(changed)[:, :5], y[:, :5])
    (gradient,) = torch.autograd.grad(y[:, :5].sum(), x)
    assert torch.count_nonzero(gradient[:, 5:]) == 0


@pytest.mark.parametrize("scales", [(0.99,), (0.99, 0.90, 0.99, 0.999)])
def test_bf16_forward_backward_and_long_sequence(scales):
    module = MatrixFastWeights(32, timescales=scales).build()
    x = torch.randn(2, 512, 32, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = module(x)
        loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )


@pytest.mark.parametrize("timescales", [(), (1.0,), (0.0,), (True,)])
def test_rejects_invalid_retention(timescales):
    with pytest.raises(ValueError):
        MatrixFastWeights(12, timescales=timescales)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_full_model_configuration(mode, monkeypatch):
    import configs.model.model as config

    monkeypatch.setattr(config, "MATRIX_MODE", mode)
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    expected = 10375680 + (6 * (384 + 384 * 64 + 64 + 384 * 8 + 8 + 8 * 384 + 64))
    if mode == "multi":
        expected += 6 * (384 * 4 + 4)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert len(model.body) == 36
    assert len(entries) == 12
