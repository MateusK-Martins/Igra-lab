from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage, MatrixCacheEntry
from src.model.composition import build_block
from src.model.definitions import LatentAttention

MODES = ("a16", "a4", "sma4", "sma4_key", "sma4_key_gelu16")


def reference(module, x):
    q = module.q_proj(x)
    outputs = []
    for t in range(x.shape[1]):
        if module.mode in ("sma4", "sma4_key", "sma4_key_gelu16"):
            matrices = module.matrix_proj(x[:, : t + 1]).reshape(
                x.shape[0], t + 1, 4, 4
            )
            responses = (matrices @ q[:, t, :, None][:, None]).squeeze(-1)
            if module.mode in ("sma4_key", "sma4_key_gelu16"):
                scores = (module.k_proj(x[:, : t + 1]) * q[:, t, None]).sum(-1) / 2
            else:
                scores = (responses * module.scoring).sum(-1) / 2
            if module.mode == "sma4_key_gelu16":
                responses = torch.nn.functional.gelu(module.response_proj(responses))
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
    if mode in ("sma4", "sma4_key", "sma4_key_gelu16"):
        assert not cache.entries
        state = cache.matrices[module.layer_id]
        assert state.matrices.numel() == 2 * 14 * 16
        assert torch.linalg.matrix_rank(state.matrices).min() == 4
        if mode in ("sma4_key", "sma4_key_gelu16"):
            assert state.keys.numel() + state.matrices.numel() == 2 * 14 * 20
        else:
            assert state.keys is None
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
    "mode,count",
    [("a16", 10399140), ("a4", 10343844), ("sma4", 10354284), ("sma4_key", 10358880)],
)
def test_full_backbone_and_parameter_count(mode, count):
    with patch("configs.model.model.MODE", mode):
        model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == count
    assert model.embedding.weight.shape == (8192, 384)
    assert len(model.body) == 24 and len(entries) == 6
    for index in range(1, 24, 8):
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
                if not any(
                    k.startswith(f"body.{i}.")
                    for i in (*range(1, 24, 8), *range(5, 24, 8))
                )
            }
        )
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])
        assert torch.equal(states[0][name], states[2][name])
        assert torch.equal(states[0][name], states[3][name])
        assert torch.equal(states[0][name], states[4][name])


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


def test_independent_keys_select_fixed_matrices():
    definition = LatentAttention(4, "sma4_key")
    module = build_block(definition, 0).double()
    matrices = torch.stack((torch.eye(4), 3 * torch.eye(4))).double()[None]
    x = torch.tensor([[[1.0, 0, 0, 0]]], dtype=torch.float64)
    with torch.no_grad():
        module.q_proj.weight.copy_(torch.eye(4))
        module.q_proj.bias.zero_()
        module.matrix_proj.weight.zero_()
        module.matrix_proj.bias.zero_()
        module.k_proj.weight.zero_()
        module.out_proj.weight.copy_(torch.eye(4))
        module.out_proj.bias.zero_()
        outputs = []
        for sign in (-1.0, 1.0):
            entry = MatrixCacheEntry(
                module.layer_id,
                matrices.clone(),
                2,
                torch.tensor(
                    [[[20 * sign, 0, 0, 0], [-20 * sign, 0, 0, 0]]], dtype=torch.float64
                ),
            )
            cache = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
            outputs.append(module(x, cache=cache))
        assert outputs[0][0, 0, 0] > 2.99
        assert outputs[1][0, 0, 0] < 1.01
    assert not hasattr(module, "scoring") and not hasattr(module, "v_proj")


def test_keyed_matrix_cache_rejects_invalid_write_without_mutation():
    definition = LatentAttention(4, "sma4_key")
    entry = definition.cache(0)
    cache = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
    matrices, keys = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4)
    cache.write_matrices(entry.layer_id, matrices, keys=keys)
    before_m, before_k = entry.matrices.clone(), entry.keys.clone()
    for invalid in (None, keys[:, :2], keys.double(), torch.randn(2, 3, 5)):
        with pytest.raises(ValueError):
            cache.write_matrices(entry.layer_id, matrices, keys=invalid)
        assert entry.count == 3
        assert torch.equal(entry.matrices, before_m) and torch.equal(
            entry.keys, before_k
        )


def test_keyed_matrix_snapshot_version_and_old_weight_incompatibility(tmp_path):
    definition = LatentAttention(4, "sma4_key")
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    path = tmp_path / "keyed.pt"
    cache.save(path)
    snapshot = torch.load(path, weights_only=True)
    assert snapshot["format_version"] == 3
    assert "keys" in snapshot["matrices"][0]
    with pytest.raises(RuntimeError):
        definition.build().load_state_dict(
            LatentAttention(4, "sma4").build().state_dict()
        )
    snapshot["matrices"][0]["keys"] = torch.empty(0, 1, 4)
    torch.save(snapshot, path)
    with pytest.raises(ValueError):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_keyed_matrix_cache_cpu_storage_cuda_reads():
    definition = LatentAttention(8, "sma4_key")
    module = build_block(definition, 0).cuda().eval()
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cuda")
    )
    x = torch.randn(2, 17, 8, device="cuda")
    with torch.no_grad():
        expected = module(x)
        actual = torch.cat([module(chunk, cache=cache) for chunk in x.split(3, 1)], 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    entry = cache.matrices[module.layer_id]
    assert entry.keys.device.type == "cpu" and entry.matrices.device.type == "cpu"
    assert entry.keys.numel() + entry.matrices.numel() == 2 * 17 * 20


def test_gelu16_parameter_count_and_aggregation_width():
    with patch("configs.model.model.MODE", "sma4_key_gelu16"):
        model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 10_372_944
    assert len(entries) == 6
    for index in range(1, 24, 8):
        module = model.body[index].branch
        assert module.q_proj.out_features == module.k_proj.out_features == 4
        assert module.response_proj.weight.shape == (16, 4)
        assert module.out_proj.weight.shape == (384, 16)


def test_nonlinearity_is_applied_before_weighted_sum():
    module = build_block(LatentAttention(4, "sma4_key_gelu16"), 0).double()
    with torch.no_grad():
        module.q_proj.weight.copy_(torch.eye(4))
        module.q_proj.bias.zero_()
        module.k_proj.weight.zero_()  # Uniform selection, independent of content.
        module.matrix_proj.weight.zero_()
        module.matrix_proj.bias.zero_()
        module.response_proj.weight.zero_()
        module.response_proj.bias.zero_()
        module.response_proj.weight[0, 0] = 1
        module.out_proj.weight.zero_()
        module.out_proj.bias.zero_()
        module.out_proj.weight[0, 0] = 1
        a = torch.zeros(4, 4, dtype=torch.float64)
        a[0, 0] = -2
        b = -a
        entry = MatrixCacheEntry(
            module.layer_id,
            torch.stack((a, b))[None],
            2,
            torch.zeros(1, 2, 4, dtype=torch.float64),
        )
        cache = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
        result = module(
            torch.tensor([[[1.0, 0, 0, 0]]], dtype=torch.float64), cache=cache
        )
        expected = torch.nn.functional.gelu(
            torch.tensor([-2.0, 2.0, 0.0], dtype=torch.float64)
        ).mean()
        torch.testing.assert_close(result[0, 0, 0], expected)
        assert result[0, 0, 0] > 0.6  # GELU(mean([-2,2,0])) would be zero.


def test_nonlinear_variant_rejects_previous_checkpoint():
    with pytest.raises(RuntimeError):
        LatentAttention(8, "sma4_key_gelu16").build().load_state_dict(
            LatentAttention(8, "sma4_key").build().state_dict()
        )
