from types import SimpleNamespace

import torch

from configs.model.model import build_model
from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheStorage
from src.model.gmlp import GMLPImpl
from src.model.no_k_variant import NoKVariantImpl
from src.model.position import rope


def test_hybrid_configuration_count_and_gate_initialization():
    model, entries = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(p.numel() for p in model.parameters()) == 12_652_960
    assert len(entries) == 8
    for i in range(6):
        m = model.body[4 * i + 1].branch
        if i % 3 == 2:
            assert isinstance(m, GQAAttentionImpl)
            assert (
                m.q_proj.weight.shape
                == m.k_proj.weight.shape
                == m.v_proj.weight.shape
                == (16, 384)
            )
            assert m.out_proj.weight.shape == (384, 16)
            assert isinstance(m.gmlp, GMLPImpl)
            assert m.gmlp.up.weight.shape == (64, 16)
            assert m.window_size == 0 and m.dropout == 0
            assert (
                m.position_rotation.func is rope
                and m.position_rotation.keywords == {"base": 10000.0}
            )
        else:
            assert isinstance(m, NoKVariantImpl) and m.normalization_scope == "global"
            torch.testing.assert_close(
                m.value_decay_gate.bias.sigmoid(), torch.linspace(0.90, 0.995, 384)
            )
            torch.testing.assert_close(
                m.write_gate.bias.sigmoid(), torch.linspace(0.20, 0.02, 384)
            )


def test_hybrid_snapshot_restore_and_mixed_cache_sizes(tmp_path):
    model, entries = build_model(SimpleNamespace(vocab_size=300))
    model.double().eval()
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    ids = torch.randint(300, (2, 17))
    with torch.no_grad():
        expected = model(ids)
        first = model(ids[:, :8], cache=cache)
        path = tmp_path / "hybrid.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        actual = torch.cat((first, model(ids[:, 8:], cache=restored)), 1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    assert len(restored.states) == 6 and len(restored.entries) == 2
    assert sum(e.value_sum.numel() for e in restored.states.values()) == 2 * (
        4 * 384 + 2 * 64 * 32
    )
    assert (
        sum(e.keys.numel() + e.values.numel() for e in restored.entries.values())
        == 2 * 2 * 17 * 32
    )


def test_hybrid_future_gradient_isolation():
    model, _ = build_model(SimpleNamespace(vocab_size=300))
    model.double()
    x = torch.randn(2, 9, 384, dtype=torch.float64, requires_grad=True)
    y = x
    for block in model.body:
        y = block(y)
    (gradient,) = torch.autograd.grad(y[:, :3].square().sum(), x)
    assert torch.count_nonzero(gradient[:, 3:]) == 0
