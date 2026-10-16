from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage, MatrixCacheEntry
from src.model.composition import build_block
from src.model.definitions import LatentAttention

MODES = ("a16", "a4", "sma4")


def reference(module, x):
    q = module.q_proj(x)
    outputs = []
    for t in range(x.shape[1]):
        if module.mode == "sma4":
            matrices = module.matrix_proj(x[:, : t + 1]).reshape(
                x.shape[0], t + 1, 4, 4
            )
            responses = (matrices @ q[:, t, :, None][:, None]).squeeze(-1)
            scores = (responses * module.scoring).sum(-1) / 2
            z = (scores.softmax(-1)[..., None] * responses).sum(1)
        else:
            k, v = module.k_proj(x[:, : t + 1]), module.v_proj(x[:, : t + 1])
            scores = (k * q[:, t, None]).sum(-1) / q.shape[-1] ** 0.5
            z = (scores.softmax(-1)[..., None] * v).sum(1)
        outputs.append(module.out_proj(z))
    return torch.stack(outputs, 1)


@pytest.mark.parametrize("mode", MODES)
def test_reference_outputs_and_all_parameter_gradients(mode):
    torch.manual_seed(42)
    module = LatentAttention(8, mode).build().double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    actual, expected = module(x), reference(module, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    variables = (x, *module.parameters())
    a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), variables)
    for ga, gb in zip(a, b, strict=True):
        torch.testing.assert_close(ga, gb, atol=1e-11, rtol=1e-10)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunk", [1, 3, 9])
def test_cache_equivalence_persistence_size_and_no_hidden_cache(mode, chunk, tmp_path):
    definition = LatentAttention(8, mode)
    module = build_block(definition, 0).double().eval()
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 13, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat([module(part, cache=cache) for part in x.split(chunk, 1)], 1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        path = tmp_path / "cache.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        next_x = torch.randn(2, 1, 8, dtype=torch.float64)
        torch.testing.assert_close(
            module(next_x, cache=restored), module(next_x, cache=cache)
        )
    assert cache.step(module.layer_id) == 14
    if mode == "sma4":
        assert not cache.entries
        state = cache.matrices[module.layer_id]
        assert state.matrices.numel() == 2 * 14 * 16
        assert torch.linalg.matrix_rank(state.matrices).min() == 4
    else:
        assert not cache.matrices
        state = cache.entries[module.layer_id]
        assert state.keys.numel() + state.values.numel() == 2 * 14 * (
            32 if mode == "a16" else 8
        )
    fresh = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    assert fresh.step(module.layer_id) == 0
    with pytest.raises(RuntimeError, match="inference-only"):
        module(x, cache=fresh)


@pytest.mark.parametrize("mode", MODES)
def test_causality_future_gradient_and_residual_once(mode):
    definition = LatentAttention(8, mode, residual=True)
    module = build_block(definition, 0).double()
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    y = module(x)
    torch.testing.assert_close(y, x + module.branch(x))
    (gradient,) = torch.autograd.grad(y[:, :3].sum(), x)
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    changed = x.detach().clone()
    changed[:, 3:] += 100
    torch.testing.assert_close(module(changed)[:, :3], y[:, :3])


@pytest.mark.parametrize(
    "mode,count", [("a16", 10375680), ("a4", 10265088), ("sma4", 10285968)]
)
def test_full_backbone_and_parameter_count(mode, count):
    with patch("configs.model.model.MODE", mode):
        model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.embedding.weight.shape == (8192, 384)
    assert len(model.body) == 24 and len(entries) == 6
    for index in range(1, 24, 4):
        assert model.body[index].branch.out_proj.out_features == 384
        assert model.body[index + 1].branch.gate.out_features == 1024


def test_shared_backbone_initial_values_identical_across_modes():
    states = []
    for mode in MODES:
        torch.manual_seed(42)
        with patch("configs.model.model.MODE", mode):
            model, _ = build_model(SimpleNamespace(vocab_size=300))
        states.append(
            {
                k: v
                for k, v in model.state_dict().items()
                if not any(k.startswith(f"body.{i}.") for i in range(1, 24, 4))
            }
        )
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])
        assert torch.equal(states[0][name], states[2][name])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_finite_forward_backward_and_long_sequence(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = LatentAttention(8, mode).build().to(device)
    x = torch.randn(2, 512, 8, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        y = module(x)
        loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(y).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )
    assert torch.isfinite(module(torch.zeros_like(x))).all()


def test_matrix_query_changes_response_without_changing_stored_matrix():
    module = LatentAttention(4, "sma4").build().double()
    with torch.no_grad():
        module.q_proj.weight.copy_(torch.eye(4))
        module.q_proj.bias.zero_()
        module.matrix_proj.weight.zero_()
        module.matrix_proj.bias.copy_(torch.eye(4).flatten())
        module.out_proj.weight.copy_(torch.eye(4))
        module.out_proj.bias.zero_()
    x = torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]], dtype=torch.float64)
    torch.testing.assert_close(module(x), x)


def test_invalid_matrix_snapshot_and_settings(tmp_path):
    for mode in ("bad", "A4"):
        with pytest.raises(ValueError):
            LatentAttention(8, mode)
    with pytest.raises(ValueError):
        LatentAttention(0, "a4")
    with pytest.raises(ValueError):
        CacheStorage(
            [MatrixCacheEntry("x", torch.zeros(1, 2, 4, 4), 1)],
            torch.device("cpu"),
            torch.device("cpu"),
        )
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format_version": 2,
            "entries": [],
            "matrices": [
                {"layer_id": "x", "matrices": torch.zeros(1, 2, 4, 4), "count": 1}
            ],
        },
        path,
    )
    with pytest.raises(ValueError):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
