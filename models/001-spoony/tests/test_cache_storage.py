import pytest
import torch

from src.model.cache_storage import CacheEntry, CacheStorage


@pytest.mark.parametrize("capacity", [1, 2, 5])
@pytest.mark.parametrize(
    "chunk_sizes", [(0, 1, 2, 7, 0, 13), (5, 5, 5), (1, 1, 1, 1, 1, 1)]
)
def test_bulk_ring_writes_match_token_by_token_reference(
    tmp_path, capacity, chunk_sizes
) -> None:
    shape = (2, 3, capacity, 4)
    entry = CacheEntry("layer", True, torch.zeros(shape), torch.zeros(shape), 0, 0)
    storage = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
    expected_keys = torch.zeros(shape)
    expected_values = torch.zeros(shape)
    pointer = count = 0
    for size in chunk_sizes:
        keys = (
            torch.arange(2 * 3 * size * 4, dtype=torch.float32).reshape(2, 3, size, 4)
            + count
        )
        values = keys + 1000
        for index in range(size):
            expected_keys[:, :, pointer, :] = keys[:, :, index, :]
            expected_values[:, :, pointer, :] = values[:, :, index, :]
            pointer = (pointer + 1) % capacity
            count += 1
        storage.write("layer", keys, values)
        torch.testing.assert_close(entry.keys, expected_keys)
        torch.testing.assert_close(entry.values, expected_values)
        assert entry.pointer == pointer
        assert storage.step("layer") == count


def test_unbounded_cache_appends_and_keeps_layers_independent(tmp_path) -> None:
    entries = [
        CacheEntry(name, False, torch.empty(1, 2, 0, 3), torch.empty(1, 2, 0, 3), 0, 0)
        for name in ("a", "b")
    ]
    storage = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    for size in (2, 3):
        storage.write("a", torch.ones(1, 2, size, 3), torch.full((1, 2, size, 3), 2.0))
    assert storage.step("a") == 5
    assert storage.step("b") == 0
    torch.testing.assert_close(entries[0].keys, torch.ones(1, 2, 5, 3))
    torch.testing.assert_close(entries[0].values, torch.full((1, 2, 5, 3), 2.0))


@pytest.mark.parametrize(
    "bounded,size", [(True, 0), (True, 2), (True, 8), (False, 0), (False, 8)]
)
def test_snapshot_restores_buffers_and_continues_writing(tmp_path, bounded, size):
    capacity = 5 if bounded else 0
    entry = CacheEntry(
        "layer",
        bounded,
        torch.zeros(1, 2, capacity, 3),
        torch.zeros(1, 2, capacity, 3),
        0,
        0,
    )
    original = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
    keys = torch.arange(6 * size, dtype=torch.float32).reshape(1, 2, size, 3)
    original.write("layer", keys, keys + 100)
    path = tmp_path / "nested" / "cache.pt"
    original.save(path)
    restored = CacheStorage.load(
        path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
    )
    for storage in (original, restored):
        storage.write("layer", torch.ones(1, 2, 3, 3), torch.full((1, 2, 3, 3), 200.0))
    actual = restored.entries["layer"]
    expected = original.entries["layer"]
    assert actual.count == expected.count
    assert actual.pointer == expected.pointer
    assert actual.bounded == expected.bounded
    torch.testing.assert_close(actual.keys, expected.keys)
    torch.testing.assert_close(actual.values, expected.values)
    torch.testing.assert_close(
        restored.read_all("layer").keys, original.read_all("layer").keys
    )
    torch.testing.assert_close(
        restored.read_all("layer").values, original.read_all("layer").values
    )


def test_snapshot_is_plain_detached_cpu_data(tmp_path):
    entry = CacheEntry(
        "layer",
        False,
        torch.ones(1, 1, 2, 3, requires_grad=True),
        torch.ones(1, 1, 2, 3, requires_grad=True),
        2,
        0,
    )
    storage = CacheStorage([entry], torch.device("cpu"), torch.device("cpu"))
    path = tmp_path / "cache.pt"
    storage.save(path)
    snapshot = torch.load(path, weights_only=True)
    assert snapshot["format_version"] == 1
    for name in ("keys", "values"):
        tensor = snapshot["entries"][0][name]
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
    assert entry.keys.requires_grad


@pytest.mark.parametrize(
    "snapshot",
    [
        {"format_version": 2, "entries": []},
        {"format_version": 1, "entries": None},
        {"format_version": 1, "entries": [{}]},
    ],
)
def test_load_rejects_invalid_snapshot(tmp_path, snapshot):
    path = tmp_path / "cache.pt"
    torch.save(snapshot, path)
    with pytest.raises((ValueError, TypeError)):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )


@pytest.mark.parametrize(
    "field,value", [("count", -1), ("pointer", 4), ("bounded", 1), ("keys", "bad")]
)
def test_load_rejects_invalid_entry(tmp_path, field, value):
    record = {
        "layer_id": "layer",
        "bounded": True,
        "keys": torch.zeros(1, 1, 5, 3),
        "values": torch.zeros(1, 1, 5, 3),
        "count": 2,
        "pointer": 2,
    }
    record[field] = value
    path = tmp_path / "cache.pt"
    torch.save({"format_version": 1, "entries": [record]}, path)
    with pytest.raises((ValueError, TypeError)):
        CacheStorage.load(
            path, storage_device=torch.device("cpu"), read_device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_snapshot_can_restore_on_a_different_device(tmp_path):
    entry = CacheEntry(
        "layer", False, torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 3), 2, 0
    )
    storage = CacheStorage([entry], torch.device("cuda"), torch.device("cuda"))
    path = tmp_path / "cache.pt"
    storage.save(path)
    restored = CacheStorage.load(
        path, storage_device=torch.device("cpu"), read_device=torch.device("cuda")
    )
    assert restored.entries["layer"].keys.device.type == "cpu"
    assert restored.read_all("layer").keys.device.type == "cuda"
    torch.testing.assert_close(
        restored.read_all("layer").keys, storage.read_all("layer").keys
    )
