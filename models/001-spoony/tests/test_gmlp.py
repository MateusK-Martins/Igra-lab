from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import GMLP


def reference(m, x):
    u, v = F.gelu(m.up(x)).chunk(2, -1)
    v = m.norm(v)
    results = []
    for t in range(x.shape[1]):
        row = t % m.window_size
        start = t - row
        gate = m.spatial_bias[row] + sum(
            m.spatial_weight[row, j] * v[:, start + j] for j in range(row + 1)
        )
        results.append(m.down(u[:, t] * gate))
    return torch.stack(results, 1)


def test_math_and_gradients_against_sequential_reference():
    torch.manual_seed(42)
    m = GMLP(4, 8, 3).build().double()
    # Nontrivial mixing makes causality/equivalence meaningful beyond initialization.
    torch.nn.init.normal_(m.spatial_weight, std=0.3)
    other = deepcopy(m)
    x = torch.randn(2, 11, 4, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    y = m(x)
    expected = reference(other, xx)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)
    y.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    assert torch.count_nonzero(m.spatial_weight.grad.triu(1)) == 0


@pytest.mark.parametrize("chunks", [[1] * 17, [2, 7, 1, 7], [17]])
def test_cache_windows_snapshot_and_constant_storage(chunks, tmp_path):
    definition = GMLP(4, 8, 3, residual=True)
    m = build_block(definition, 0).double().eval()
    torch.nn.init.normal_(m.branch.spatial_weight, std=0.3)
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 17, 4, dtype=torch.float64)
    with torch.no_grad():
        expected = x + reference(m.branch, x)
        pieces = []
        start = 0
        for n in chunks:
            pieces.append(m(x[:, start : start + n], cache=cache))
            start += n
            cache.save(tmp_path / "cache.pt")
            cache = CacheStorage.load(
                tmp_path / "cache.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
            state = next(iter(cache.states.values()))
            assert state.value_sum.numel() == 2 * 3 * 4
            assert state.count == start
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-12, rtol=1e-12)


def test_causality_and_gradcheck():
    m = GMLP(2, 4, 3).build().double()
    x = torch.randn(1, 7, 2, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(m(x)[:, :2].sum(), x)
    assert torch.count_nonzero(grad[:, 2:]) == 0
    assert torch.autograd.gradcheck(m, (x,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_and_long_sequence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = GMLP(8, 16, 64).build().to(device)
    x = torch.zeros(2, 1025, 8, device=device, requires_grad=True)
    with torch.autocast(device, dtype=torch.bfloat16):
        y = m(x)
        loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(y).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert torch.equal(m.spatial_bias, torch.ones_like(m.spatial_bias))


@pytest.mark.parametrize(
    "kwargs", [{"hidden_features": 3}, {"window_size": 0}, {"residual": 1}]
)
def test_invalid_definition(kwargs):
    with pytest.raises(ValueError):
        GMLP(4, **kwargs)
