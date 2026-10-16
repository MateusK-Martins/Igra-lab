import pytest
import torch

from src.model.cache_storage import CacheStorage
from src.model.composition import build_block
from src.model.definitions import GMLP


def test_256_window_boundary_cache_and_future_gradients(tmp_path):
    d = GMLP(4, 8, 256, residual=True)
    m = build_block(d, 0).double()
    torch.nn.init.normal_(m.branch.spatial_weight, std=0.03)
    x = torch.randn(1, 513, 4, dtype=torch.float64, requires_grad=True)
    y = m(x)
    (gradient,) = torch.autograd.grad(y[:, :256].square().sum(), x)
    assert not gradient[:, 256:].count_nonzero()
    # Fixed-window SGU resets its temporal history at256; the second window
    # must not receive gradients from the first through this mixer alone.
    (gradient,) = torch.autograd.grad(m(x)[:, 256:512].sum(), x)
    assert not gradient[:, :256].count_nonzero()
    c = CacheStorage([d.cache(0)], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        pieces = []
        for part in [x[:, :255], x[:, 255:257], x[:, 257:511], x[:, 511:]]:
            pieces.append(m(part, cache=c))
            c.save(tmp_path / "long.pt")
            c = CacheStorage.load(
                tmp_path / "long.pt",
                storage_device=torch.device("cpu"),
                read_device=torch.device("cpu"),
            )
    torch.testing.assert_close(torch.cat(pieces, 1), y, atol=1e-12, rtol=1e-12)
    assert c.step(m.branch.layer_id) == 513
    assert next(iter(c.states.values())).value_sum.numel() == 256 * 4
    assert m.branch.spatial_weight.shape == (256, 256)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_256_window_bf16_training_and_cached_chunks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    d = GMLP(4, 8, 256, residual=True)
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
        cached = torch.cat([m(part, cache=c) for part in x.split(63, 1)], 1)
    torch.testing.assert_close(cached, y, atol=0.025, rtol=0.03)
