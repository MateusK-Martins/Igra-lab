"""Independent scalar-angle references for relative-position matrix retrieval."""

import pytest
import torch
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import LatentAttention

MODES = ("a16", "a4", "sma4", "sma4_key", "sma4_key_gelu16")


def rotation(width, position, *, base=10000.0):
    matrix = torch.zeros(width, width, dtype=torch.float64)
    for index in range(0, width, 2):
        angle = torch.tensor(position / base ** (index / width), dtype=torch.float64)
        c, s = angle.cos(), angle.sin()
        matrix[index, index] = c
        matrix[index, index + 1] = -s
        matrix[index + 1, index] = s
        matrix[index + 1, index + 1] = c
    return matrix


def reference(module, x, offset=0):
    q = module.q_proj(x)
    outputs = []
    for t in range(x.shape[1]):
        query = q[:, t] @ rotation(q.shape[-1], t + offset).T
        scores, values = [], []
        for i in range(t + 1):
            if module.mode.startswith("sma"):
                m = (
                    module.matrix_proj(x[:, i]).reshape(-1, 4, 4)
                    @ rotation(4, i + offset).T
                )
                response = (m @ query[..., None]).squeeze(-1)
                if module.mode == "sma4":
                    score = (response * module.scoring).sum(-1) / 2
                else:
                    key = module.k_proj(x[:, i]) @ rotation(4, i + offset).T
                    score = (key * query).sum(-1) / 2
                if module.mode == "sma4_key_gelu16":
                    response = F.gelu(module.response_proj(response))
                values.append(response)
            else:
                key = module.k_proj(x[:, i]) @ rotation(q.shape[-1], i + offset).T
                score = (query * key).sum(-1) / q.shape[-1] ** 0.5
                values.append(module.v_proj(x[:, i]))
            scores.append(score)
        weights = torch.stack(scores, 1).softmax(1)
        z = (weights[..., None] * torch.stack(values, 1)).sum(1)
        outputs.append(module.out_proj(z))
    return torch.stack(outputs, 1)


@pytest.mark.parametrize("mode", MODES)
def test_rotary_reference_all_gradients_and_relative_shift(mode):
    module = LatentAttention(8, mode, rope_base=10000.0).build().double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    actual, expected = module(x), reference(module, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        expected, reference(module, x, offset=37), atol=1e-12, rtol=1e-12
    )
    args = (x, *module.parameters())
    a = torch.autograd.grad(actual.square().sum(), args, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), args)
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, atol=1e-11, rtol=1e-10)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunk", [1, 4])
def test_rotary_cache_offsets_snapshot_causality_and_size(mode, chunk, tmp_path):
    d = LatentAttention(8, mode, rope_base=10000.0)
    m = build_block(d, 0).double().eval()
    cache = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    x = torch.randn(2, 13, 8, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(m(x)[:, :4].sum(), x)
    assert torch.count_nonzero(gradient[:, 4:]) == 0
    with torch.no_grad():
        expected = m(x)
        parts = [m(part, cache=cache) for part in x[:, :9].split(chunk, 1)]
        path = tmp_path / "rope.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        parts.append(m(x[:, 9:], cache=restored))
        torch.testing.assert_close(
            torch.cat(parts, 1), expected, atol=1e-12, rtol=1e-12
        )
    assert restored.step(m.layer_id) == 13
    assert "-rope-" in m.layer_id
    if mode.startswith("sma"):
        entry = restored.matrices[m.layer_id]
        assert entry.matrices.numel() + (
            0 if entry.keys is None else entry.keys.numel()
        ) == 2 * 13 * (16 if mode == "sma4" else 20)
    else:
        entry = restored.entries[m.layer_id]
        assert entry.keys.numel() + entry.values.numel() == 2 * 13 * (
            32 if mode == "a16" else 8
        )


@pytest.mark.parametrize("mode", MODES)
def test_rope_does_not_change_initial_weights_or_parameter_count(mode):
    torch.manual_seed(42)
    plain = LatentAttention(8, mode).build()
    torch.manual_seed(42)
    rotary = LatentAttention(8, mode, rope_base=10000.0).build()
    assert plain.state_dict().keys() == rotary.state_dict().keys()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, rotary.state_dict()[name])


@pytest.mark.parametrize("mode", MODES)
def test_rotary_cuda_bf16_backward_and_cpu_cache(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    d = LatentAttention(8, mode, rope_base=10000.0)
    m = build_block(d, 0).cuda()
    x = torch.randn(2, 64, 8, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    cache = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cached = torch.cat([m(part, cache=cache) for part in x.split(7, 1)], 1)
    torch.testing.assert_close(cached, y, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("base", [True, 0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_rotary_base(base):
    with pytest.raises(ValueError, match="rope_base"):
        LatentAttention(8, "a4", rope_base=base)
