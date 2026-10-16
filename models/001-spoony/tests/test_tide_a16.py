from types import SimpleNamespace

import torch

from configs.model.model import build_model
from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheStorage
from src.model.gmlp import GMLPImpl
from src.model.no_k_variant import NoKVariantImpl
from src.model.position import rope


def test_configuration_parameter_budget_and_mechanisms():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 10_410_720
    assert len(entries) == 12
    assert not any(isinstance(m, NoKVariantImpl) for m in model.modules())
    attention = [m for m in model.modules() if isinstance(m, GQAAttentionImpl)]
    gmlps = [m for m in model.modules() if isinstance(m, GMLPImpl)]
    assert len(attention) == len(gmlps) == 6
    for m in attention:
        assert (
            m.q_proj.weight.shape
            == m.k_proj.weight.shape
            == m.v_proj.weight.shape
            == (16, 384)
        )
        assert m.out_proj.weight.shape == (384, 16)
        assert isinstance(m.gmlp, GMLPImpl) and m.gmlp.up.weight.shape == (64, 16)
        assert m.gmlp.window_size == 64
        assert m.position_rotation.func is rope and m.position_rotation.keywords == {
            "base": 10000.0
        }


def test_snapshot_restore_and_cache_sizes(tmp_path):
    model, entries = build_model(SimpleNamespace(vocab_size=300))
    model.double().eval()
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    ids = torch.randint(300, (2, 81))
    with torch.no_grad():
        expected = model(ids)
        first = model(ids[:, :63], cache=cache)
        cache.save(tmp_path / "cache.pt")
        restored = CacheStorage.load(
            tmp_path / "cache.pt",
            storage_device=torch.device("cpu"),
            read_device=torch.device("cpu"),
        )
        actual = torch.cat((first, model(ids[:, 63:], cache=restored)), 1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    assert len(restored.states) == len(restored.entries) == 6
    assert sum(e.value_sum.numel() for e in restored.states.values()) == 2 * 6 * 2048
    assert (
        sum(e.keys.numel() + e.values.numel() for e in restored.entries.values())
        == 2 * 6 * 81 * 32
    )


def test_future_gradient_isolation():
    model, _ = build_model(SimpleNamespace(vocab_size=300))
    model.double()
    x = torch.randn(2, 9, 384, dtype=torch.float64, requires_grad=True)
    y = x
    for block in model.body:
        y = block(y)
    (gradient,) = torch.autograd.grad(y[:, :3].square().sum(), x)
    assert not gradient[:, 3:].count_nonzero()
