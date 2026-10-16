"""Mixed attention/recurrent composition, causal gradients and cache persistence."""

from types import SimpleNamespace

import torch

from configs.model.model import build_model
from src.model.cache_storage import CacheStorage


def test_hybrid_future_gradient_isolation():
    torch.manual_seed(42)
    model, _ = build_model(SimpleNamespace(vocab_size=31))
    tokens = torch.randint(0, 31, (2, 9))
    captured = []

    def capture(module, args, output):
        output.retain_grad()
        captured.append(output)

    handle = model.embedding.register_forward_hook(capture)
    logits = model(tokens)
    handle.remove()
    logits[:, :4].sum().backward()
    assert torch.count_nonzero(captured[0].grad[:, 4:]) == 0
    altered = tokens.clone()
    altered[:, 4:] = (altered[:, 4:] + 1) % 31
    with torch.no_grad():
        torch.testing.assert_close(model(altered)[:, :4], logits[:, :4])


def test_mixed_cache_snapshot_continuation(tmp_path):
    model, entries = build_model(SimpleNamespace(vocab_size=31))
    model.double().eval()
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    tokens = torch.randint(0, 31, (2, 13))
    with torch.no_grad():
        full = model(tokens)
        prefix = model(tokens[:, :7], cache=cache)
        path = tmp_path / "mixed.pt"
        cache.save(path)
        restored = CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )
        suffix = model(tokens[:, 7:], cache=restored)
    assert len(restored.states) == 6 and len(restored.entries) == 2
    torch.testing.assert_close(
        torch.cat((prefix, suffix), 1), full, rtol=1e-7, atol=1e-9
    )
