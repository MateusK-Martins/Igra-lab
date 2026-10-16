"""Recurrent cache persistence, validation and device transfers."""

import pytest
import torch

from src.model.cache_storage import CacheStorage, StateEntry


def storage(device="cpu", read="cpu"):
    return CacheStorage(
        [StateEntry("layer", torch.empty(0, 1, 0), torch.empty(0, 1, 4))],
        torch.device(device),
        torch.device(read),
    )


def test_write_snapshot_restore_and_continue(tmp_path):
    cache = storage()
    state = torch.randn(2, 1, 4, requires_grad=True)
    cache.write_state("layer", torch.empty(2, 1, 0), state, 7)
    assert cache.step("layer") == 7
    assert not cache.read_state("layer").value_sum.requires_grad
    path = tmp_path / "cache.pt"
    cache.save(path)
    restored = CacheStorage.load(
        path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
    )
    torch.testing.assert_close(restored.read_state("layer").value_sum, state)
    for item in (cache, restored):
        item.write_state("layer", torch.empty(2, 1, 0), state.detach() + 1, 3)
        assert item.step("layer") == 10
    torch.testing.assert_close(
        cache.read_state("layer").value_sum, restored.read_state("layer").value_sum
    )


def test_invalid_write_does_not_mutate_cache():
    cache = storage()
    previous = cache.states["layer"].value_sum
    with pytest.raises(ValueError):
        cache.write_state("layer", torch.empty(2, 1, 0), torch.ones(2, 1, 5), 1)
    assert cache.states["layer"].value_sum is previous
    assert cache.step("layer") == 0


@pytest.mark.parametrize(
    "snapshot",
    [
        {"format_version": 2, "entries": []},
        {"format_version": 1, "entries": None},
        {"format_version": 1, "entries": [{}]},
        {"format_version": 1, "entries": [], "states": [{}]},
    ],
)
def test_reject_invalid_or_attention_snapshot(tmp_path, snapshot):
    path = tmp_path / "bad.pt"
    torch.save(snapshot, path)
    with pytest.raises((ValueError, TypeError)):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )


def test_duplicate_layers_rejected():
    entry = StateEntry("layer", torch.empty(0, 1, 0), torch.empty(0, 1, 4))
    with pytest.raises(ValueError, match="Duplicate"):
        CacheStorage([entry, entry], torch.device("cpu"), torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_transfer_and_snapshot(tmp_path):
    cache = storage("cpu", "cuda")
    state = torch.randn(2, 1, 4, device="cuda")
    cache.write_state("layer", state.new_empty(2, 1, 0), state, 2)
    assert cache.states["layer"].value_sum.device.type == "cpu"
    torch.testing.assert_close(cache.read_state("layer").value_sum, state)
    path = tmp_path / "cache.pt"
    cache.save(path)
    restored = CacheStorage.load(
        path, storage_device=torch.device("cuda"), read_device=torch.device("cpu")
    )
    assert restored.states["layer"].value_sum.device.type == "cuda"
    torch.testing.assert_close(restored.read_state("layer").value_sum, state.cpu())
