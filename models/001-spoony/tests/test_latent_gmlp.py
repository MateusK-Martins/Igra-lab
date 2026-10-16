from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import GQAAttention
from src.model.position import rope


def definition(**kwargs):
    args = {
        "input_features": 6,
        "output_features": 6,
        "num_query_heads": 1,
        "num_kv_heads": 1,
        "head_features": 4,
        "window_size": 0,
        "dropout": 0.0,
        "position_rotation": partial(rope, base=10000.0),
        "gmlp_hidden_features": 16,
        "gmlp_window_size": 4,
        "residual": True,
    }
    args.update(kwargs)
    return GQAAttention(**args)


def reference(m, x):
    positions = torch.arange(x.shape[1], device=x.device)
    q = m.position_rotation(m.q_proj(x).unsqueeze(1), positions)[:, 0]
    k = m.position_rotation(m.k_proj(x).unsqueeze(1), positions)[:, 0]
    v = m.v_proj(x)
    retrieved = []
    for t in range(x.shape[1]):
        scores = (q[:, t, None] * k[:, : t + 1]).sum(-1) / 2
        retrieved.append((scores.softmax(-1)[..., None] * v[:, : t + 1]).sum(1))
    z = torch.stack(retrieved, 1)
    g = m.gmlp
    a, b = F.gelu(g.up(z)).chunk(2, -1)
    b = g.norm(b)
    mixed = []
    for t in range(z.shape[1]):
        row = t % g.window_size
        gate = g.spatial_bias[row] + sum(
            g.spatial_weight[row, j] * b[:, t - row + j] for j in range(row + 1)
        )
        mixed.append(g.down(a[:, t] * gate))
    return x + m.out_proj(z + torch.stack(mixed, 1))


def test_compressed_after_retrieval_math_and_gradients():
    m = build_block(definition(), 0).double()
    torch.nn.init.normal_(m.branch.gmlp.spatial_weight, std=0.3)
    other = deepcopy(m)
    x = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    actual = m(x)
    expected = reference(other.branch, xx)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-9)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 17])
def test_full_chunked_cache_and_snapshot(chunk_size, tmp_path):
    d = definition()
    m = build_block(d, 0).double().eval()
    torch.nn.init.normal_(m.branch.gmlp.spatial_weight, std=0.3)
    c = CacheStorage(d.cache(0), torch.device("cpu"), torch.device("cpu"))
    x = torch.randn(2, 17, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = m(x)
        parts = []
        for part in x.split(chunk_size, 1):
            parts.append(m(part, cache=c))
            c.save(tmp_path / "cache.pt")
            c = CacheStorage.load(
                tmp_path / "cache.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=1e-12, rtol=1e-12)
    assert c.step(m.branch.layer_id) == c.step(m.branch.gmlp.layer_id) == 17
    assert next(iter(c.states.values())).value_sum.numel() == 2 * 4 * 8
    assert next(iter(c.entries.values())).keys.shape == (2, 1, 17, 4)


def test_causality_and_joint_gradcheck():
    m = build_block(definition(), 0).double()
    x = torch.randn(1, 5, 6, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(m(x)[:, :2].sum(), x)
    assert not grad[:, 2:].count_nonzero()
    assert torch.autograd.gradcheck(m, (x,))


def test_zero_gmlp_recovers_original_attention():
    m = build_block(definition(), 0).double()
    with torch.no_grad():
        m.branch.gmlp.down.weight.zero_()
        m.branch.gmlp.down.bias.zero_()
    old = build_block(definition(gmlp_hidden_features=0), 0).double()
    old.load_state_dict({k: v for k, v in m.state_dict().items() if ".gmlp." not in k})
    x = torch.randn(2, 9, 6, dtype=torch.float64)
    torch.testing.assert_close(m(x), old(x), atol=0, rtol=0)


def test_bad_cache_position_rejected_before_write():
    d = definition()
    m = build_block(d, 0)
    c = CacheStorage(d.cache(0), torch.device("cpu"), torch.device("cpu"))
    c.states[m.branch.gmlp.layer_id].count = 1
    with torch.no_grad(), pytest.raises(ValueError, match="positions disagree"):
        m(torch.randn(1, 1, 6), cache=c)
    assert c.step(m.branch.layer_id) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_training_and_cached_inference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    d = definition()
    m = build_block(d, 0).to(device)
    x = torch.randn(2, 33, 6, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        y = m(x)
        y.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
    c = CacheStorage(d.cache(0), torch.device("cpu"), torch.device(device))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        expected = m(x)
        cached = torch.cat([m(p, cache=c) for p in x.split(3, 1)], 1)
    torch.testing.assert_close(cached, expected, atol=0.025, rtol=0.03)


@pytest.mark.parametrize("value", [-2, 3, True, 1.5])
def test_invalid_expansion(value):
    with pytest.raises(ValueError):
        definition(gmlp_hidden_features=value)
