from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import GMLP, GQAAttention


def reference(m, x):
    u, v = F.gelu(m.up(x)).chunk(2, -1)
    v = m.norm(v)
    dh = m.channels // m.num_heads
    results = []
    for t in range(x.shape[1]):
        row = t % m.window_size
        start = t - row
        groups = []
        for h in range(m.num_heads):
            sl = slice(h * dh, (h + 1) * dh)
            gate = m.spatial_bias[h, row] + sum(
                m.spatial_weight[h, row, j] * v[:, start + j, sl]
                for j in range(row + 1)
            )
            groups.append(u[:, t, sl] * gate)
        results.append(m.down(torch.cat(groups, -1)))
    return torch.stack(results, 1)


def test_independent_head_reference_and_gradients():
    m = GMLP(4, 8, 3, num_heads=2).build().double()
    torch.nn.init.normal_(m.spatial_weight, std=0.3)
    other = deepcopy(m)
    x = torch.randn(2, 8, 4, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    actual = m(x)
    expected = reference(other, xx)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    assert all(m.spatial_weight.grad[h].abs().sum() > 0 for h in range(2))
    assert not m.spatial_weight.grad.triu(1).count_nonzero()
    assert torch.autograd.gradcheck(m, (x,))


def test_identical_head_filters_recover_single_head_with_shared_norm():
    one = GMLP(4, 8, 3).build().double()
    two = GMLP(4, 8, 3, num_heads=2).build().double()
    state = one.state_dict()
    state["spatial_weight"] = state["spatial_weight"][None].repeat(2, 1, 1)
    state["spatial_bias"] = state["spatial_bias"][None].repeat(2, 1)
    two.load_state_dict(state)
    x = torch.randn(2, 8, 4, dtype=torch.float64, requires_grad=True)
    actual = two(x)
    expected = one(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(two.spatial_weight.grad.sum(0), one.spatial_weight.grad)
    torch.testing.assert_close(two.spatial_bias.grad.sum(0), one.spatial_bias.grad)
    for name in ["up", "norm", "down"]:
        for a, b in zip(
            getattr(one, name).parameters(),
            getattr(two, name).parameters(),
            strict=True,
        ):
            torch.testing.assert_close(a.grad, b.grad)
    assert two.norm.normalized_shape == (4,)


def test_temporal_filters_independent_before_down_projection():
    m = GMLP(4, 8, 3, num_heads=2).build().double()
    with torch.no_grad():
        m.down.weight[:, 2:].zero_()
    x = torch.randn(2, 7, 4, dtype=torch.float64)
    first = m(x)
    first.sum().backward()
    assert m.spatial_weight.grad[0].abs().sum() > 0
    assert not m.spatial_weight.grad[1].count_nonzero()
    with torch.no_grad():
        m.spatial_weight[1].add_(10)
        m.spatial_bias[1].add_(10)
    torch.testing.assert_close(m(x), first, atol=0, rtol=0)


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_heads_cache_snapshot_and_future_isolation(chunk, tmp_path):
    d = GMLP(4, 8, 4, residual=True, num_heads=2)
    m = build_block(d, 0).double()
    torch.nn.init.normal_(m.branch.spatial_weight, std=0.3)
    x = torch.randn(2, 17, 4, dtype=torch.float64, requires_grad=True)
    y = m(x)
    (gradient,) = torch.autograd.grad(y[:, :3].sum(), x)
    assert not gradient[:, 3:].count_nonzero()
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        parts = []
        for part in x.split(chunk, 1):
            parts.append(m(part, cache=c))
            c.save(tmp_path / "heads.pt")
            c = CacheStorage.load(
                tmp_path / "heads.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(parts, 1), y, atol=1e-12, rtol=1e-12)
    assert next(iter(c.states.values())).value_sum.numel() == 2 * 4 * 4
    assert "heads2" in m.branch.layer_id


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_heads_256_window_bf16_long_sequence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    d = GMLP(4, 8, 256, residual=True, num_heads=2)
    m = build_block(d, 0).to(device)
    x = torch.randn(2, 513, 4, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        y = m(x)
        y.square().mean().backward()
    assert y.isfinite().all() and all(
        p.grad is not None and p.grad.isfinite().all() for p in m.parameters()
    )
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device(device))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        actual = torch.cat([m(p, cache=c) for p in x.split(63, 1)], 1)
    torch.testing.assert_close(actual, y, atol=0.025, rtol=0.03)


@pytest.mark.parametrize("heads", [0, -1, 3, True])
def test_invalid_head_count(heads):
    with pytest.raises(ValueError):
        GMLP(4, 8, 3, num_heads=heads)


def test_old_checkpoint_shapes_not_reinterpreted():
    one = GMLP(4, 8, 3).build()
    two = GMLP(4, 8, 3, num_heads=2).build()
    with pytest.raises(RuntimeError):
        two.load_state_dict(one.state_dict())


def test_attention_rejects_invalid_gmlp_heads():
    from functools import partial

    from src.model.position import rope

    args = {
        "input_features": 4,
        "output_features": 4,
        "num_query_heads": 1,
        "num_kv_heads": 1,
        "head_features": 4,
        "window_size": 0,
        "dropout": 0,
        "position_rotation": partial(rope, base=10000.0),
    }
    with pytest.raises(ValueError):
        GQAAttention(**args, gmlp_hidden_features=8, gmlp_num_heads=3)
    with pytest.raises(ValueError):
        GQAAttention(**args, gmlp_num_heads=2)
