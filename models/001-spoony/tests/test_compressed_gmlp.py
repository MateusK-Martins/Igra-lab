from copy import deepcopy

import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import CompressedGMLP


def test_independent_compressed_branch_reference_and_gradients():
    d = CompressedGMLP(6, 2, 8, 4, residual=True)
    m = build_block(d, 0).double()
    other = deepcopy(m)
    x = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    z = other.branch.input_proj(xx)
    expected = xx + other.branch.output_proj(z + other.branch.gmlp(z))
    actual = m(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    assert torch.autograd.gradcheck(m, (x,))


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_compressed_cache_snapshot_causality_and_budget(chunk_size, tmp_path):
    d = CompressedGMLP(6, 2, 8, 4, residual=True)
    m = build_block(d, 0).double()
    torch.nn.init.normal_(m.branch.gmlp.spatial_weight, std=0.3)
    x = torch.randn(2, 17, 6, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(m(x)[:, :3].sum(), x)
    assert not gradient[:, 3:].count_nonzero()
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        expected = m(x)
        pieces = []
        for part in x.split(chunk_size, 1):
            pieces.append(m(part, cache=c))
            c.save(tmp_path / "state.pt")
            c = CacheStorage.load(
                tmp_path / "state.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-12, rtol=1e-12)
    assert not c.entries and c.step(m.branch.layer_id) == 17
    assert next(iter(c.states.values())).value_sum.numel() == 2 * 4 * 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"latent_features": 0},
        {"hidden_features": 3},
        {"window_size": 0},
    ],
)
def test_invalid_compressed_dimensions(kwargs):
    with pytest.raises(ValueError):
        CompressedGMLP(**({"input_features": 6} | kwargs))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_compressed_branch_training_and_cache(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    d = CompressedGMLP(8, 4, 16, 4, residual=True)
    m = build_block(d, 0).to(device)
    x = torch.randn(2, 17, 8, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        m(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device(device))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        full = m(x)
        cached = torch.cat([m(part, cache=c) for part in x.split(3, 1)], 1)
    torch.testing.assert_close(cached, full, atol=0.025, rtol=0.03)
