"""Independent window references, scale routing, gradients and streaming history."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage, HistoryEntry
from src.model.composition import build_block
from src.model.definitions import TemporalDifferenceMixer


def direct(x, scales):
    output = []
    for t in range(x.shape[1]):
        row = []
        for m in scales:
            recent = x[:, max(0, t - m + 1) : t + 1].sum(dim=1) / m
            previous = x[:, max(0, t - 2 * m + 1) : max(0, t - m + 1)].sum(dim=1) / m
            row.append(recent - previous)
        output.append(torch.stack(row, dim=1))
    return torch.stack(output, dim=1)


def explicit(m, x):
    differences = direct(x, m.scales)
    projected = torch.stack(
        [
            F.linear(differences[:, :, i], m.scale_weights[i])
            for i in range(len(m.scales))
        ],
        dim=2,
    )
    alpha = m.router(x).softmax(-1)
    return m.out_proj((alpha[..., None] * projected).sum(2))


@pytest.mark.parametrize("length", [1, 7, 65, 137])
def test_differences_match_direct_averages(length):
    m = TemporalDifferenceMixer(4).build().double()
    x = torch.randn(2, length, 4, dtype=torch.float64)
    torch.testing.assert_close(
        m.temporal_differences(x), direct(x, m.scales), atol=1e-12, rtol=1e-12
    )


def test_constant_step_ramp_and_zero_padding():
    m = TemporalDifferenceMixer(2).build().double()
    constant = torch.ones(1, 192, 2, dtype=torch.float64)
    differences = m.temporal_differences(constant)
    torch.testing.assert_close(
        differences[:, 63:], torch.zeros_like(differences[:, 63:]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        differences[:, 0, :, 0],
        torch.tensor([[1 / s for s in m.scales]], dtype=torch.float64),
    )
    step = torch.zeros_like(constant)
    step[:, 64:] = 1
    torch.testing.assert_close(m.temporal_differences(step), direct(step, m.scales))
    ramp = torch.arange(192, dtype=torch.float64)[None, :, None].expand(1, -1, 2)
    torch.testing.assert_close(
        m.temporal_differences(ramp)[:, 63:, :, 0],
        torch.tensor(m.scales, dtype=torch.float64).expand(1, 129, 6),
    )


def test_readout_gradients_and_gradcheck():
    m = TemporalDifferenceMixer(2, scales=(1, 2, 3)).build().double()
    with torch.no_grad():
        m.router.weight.normal_(0, 0.2)
    x = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    actual = m(x)
    expected = explicit(m, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    incoming = torch.randn_like(actual)
    targets = (x, *m.parameters())
    for a, e in zip(
        torch.autograd.grad(actual, targets, incoming),
        torch.autograd.grad(expected, targets, incoming),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(m, (x,))


def test_uniform_initial_router_and_learned_input_dependence():
    m = TemporalDifferenceMixer(4).build()
    x = torch.randn(2, 9, 4, requires_grad=True)
    torch.testing.assert_close(m.router(x).softmax(-1), torch.full((2, 9, 6), 1 / 6))
    m(x).square().sum().backward()
    assert m.router.weight.grad.abs().sum() > 0
    for p in m.parameters():
        assert torch.isfinite(p.grad).all()
    with torch.no_grad():
        m.router.weight.normal_()
    assert not torch.equal(m.router(x[:, 0]).softmax(-1), m.router(x[:, 1]).softmax(-1))
    assert sum(p.numel() for p in m.parameters()) == 7 * 4 * 4 + 6 * 4


@pytest.mark.parametrize("chunk", [1, 7, 80])
def test_streaming_snapshot_and_constant_history(tmp_path, chunk):
    definition = TemporalDifferenceMixer(4)
    m = build_block(definition, 0).double()
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    x = torch.randn(2, 197, 4, dtype=torch.float64)
    with torch.no_grad():
        full = m(x)
        pieces = []
        for start in range(0, 197, chunk):
            pieces.append(m(x[:, start : start + chunk], cache=cache))
            entry = cache.histories[m.layer_id]
            assert entry.values.shape[1] == min(entry.count, 63)
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-12, rtol=1e-12)
        assert entry.count == 197 and entry.values.numel() == 2 * 63 * 4
        path = tmp_path / "tdm.pt"
        cache.save(path)
        assert torch.load(path, weights_only=True)["format_version"] == 2
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        more = torch.randn(2, 11, 4, dtype=torch.float64)
        torch.testing.assert_close(m(more, cache=cache), m(more, cache=restored))
        fresh = CacheStorage(
            [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
        )
        torch.testing.assert_close(m(x, cache=fresh), full)


def test_causality_residual_and_no_cached_training():
    definition = TemporalDifferenceMixer(4, residual=True)
    m = build_block(definition, 0).double()
    x = torch.randn(2, 89, 4, dtype=torch.float64, requires_grad=True)
    out = m(x)
    torch.testing.assert_close(out, x + m.branch(x))
    grad = torch.autograd.grad(out[:, :17].sum(), x)[0]
    assert torch.count_nonzero(grad[:, 17:]) == 0
    altered = x.detach().clone()
    altered[:, 17:] += 100
    torch.testing.assert_close(out[:, :17], m(altered)[:, :17])
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="no_grad"):
        m(x, cache=cache)


@pytest.mark.parametrize("shape", [(2, 0, 4), (0, 7, 4)])
def test_empty(shape):
    m = TemporalDifferenceMixer(4).build()
    x = torch.empty(shape, requires_grad=True)
    out = m(x)
    assert out.shape == shape
    out.sum().backward()
    assert x.grad.shape == shape


def test_long_sequence_finite():
    m = TemporalDifferenceMixer(4).build()
    x = torch.randn(1, 4096, 4, requires_grad=True)
    out = m(x)
    out.square().mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("scales", [(), (0,), (-1,), (2, 1), (1, 1), (True,), [1, 2]])
def test_invalid_scales(scales):
    with pytest.raises(ValueError, match="scales"):
        TemporalDifferenceMixer(4, scales=scales)


def test_reject_bad_history_without_mutation(tmp_path):
    entry = HistoryEntry("tdm", torch.empty(0, 0, 4), 63)
    cache = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
    original = entry.values
    with pytest.raises(ValueError):
        cache.write_history("tdm", torch.zeros(2, 7, 5))
    assert entry.values is original and entry.count == 0
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format_version": 2,
            "entries": [],
            "states": [],
            "histories": [
                {
                    "layer_id": "tdm",
                    "values": torch.empty(1, 64, 4),
                    "capacity": 63,
                    "count": 64,
                }
            ],
        },
        path,
    )
    with pytest.raises(ValueError):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_forward_backward_and_cpu_storage():
    definition = TemporalDifferenceMixer(16)
    m = build_block(definition, 0).cuda()
    x = torch.randn(2, 137, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x)
        out.float().square().mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert torch.isfinite(p.grad).all()
    low = x.detach().bfloat16()
    assert m.temporal_differences(low).dtype == torch.float32
    expected = direct(low.float(), m.scales)
    torch.testing.assert_close(
        m.temporal_differences(low), expected, atol=2e-5, rtol=2e-5
    )
    cache = CacheStorage(
        [definition.cache(0)], torch.device("cpu"), torch.device("cuda")
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = m(x)
        chunks = torch.cat([m(x[:, :67], cache=cache), m(x[:, 67:], cache=cache)], 1)
    torch.testing.assert_close(chunks, full, atol=0.005, rtol=0.02)
    assert cache.histories[m.layer_id].values.device.type == "cpu"


def test_full_model_budget_and_memory_layout():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 13_481_856
    assert sum(isinstance(e, HistoryEntry) for e in entries) == 4
    assert len(entries) == 8
    assert len(model.body) == 25
