from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass()
class CacheEntry:
    layer_id: str
    bounded: bool
    keys: Tensor
    values: Tensor
    count: int
    pointer: int


class CacheStorage:
    def __init__(
        self,
        entries: list[CacheEntry],
        storage_device: torch.device,
        read_device: torch.device,
    ) -> None:
        self.storage_device = storage_device
        self.read_device = read_device

        self.entries: dict[str, CacheEntry] = {}

        for entry in entries:
            if entry.layer_id in self.entries:
                raise ValueError(f"Duplicate cache layer_id: {entry.layer_id!r}")

            entry.keys = entry.keys.to(self.storage_device)
            entry.values = entry.values.to(self.storage_device)

            self.entries[entry.layer_id] = entry

    def save(self, path: Path) -> None:
        snapshot = {
            "format_version": 1,
            "entries": [
                {
                    "layer_id": entry.layer_id,
                    "bounded": entry.bounded,
                    "keys": entry.keys.detach().cpu(),
                    "values": entry.values.detach().cpu(),
                    "count": entry.count,
                    "pointer": entry.pointer,
                }
                for entry in self.entries.values()
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        storage_device: torch.device,
        read_device: torch.device,
    ) -> "CacheStorage":
        snapshot = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(snapshot, dict) or snapshot.get("format_version") != 1:
            raise ValueError("Unsupported cache snapshot format_version")
        records = snapshot.get("entries")
        if not isinstance(records, list):
            raise TypeError("Cache snapshot entries must be a list")

        entries = []
        fields = {"layer_id", "bounded", "keys", "values", "count", "pointer"}
        for record in records:
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError("Invalid cache snapshot entry fields")
            entry = CacheEntry(**record)
            if not isinstance(entry.layer_id, str) or not entry.layer_id:
                raise ValueError("Cache layer_id must be a nonempty string")
            if type(entry.bounded) is not bool:
                raise ValueError("Cache bounded must be a boolean")
            if (
                not isinstance(entry.keys, Tensor)
                or not isinstance(entry.values, Tensor)
                or entry.keys.ndim != 4
                or entry.keys.shape != entry.values.shape
                or entry.keys.dtype != entry.values.dtype
            ):
                raise ValueError("Invalid cache snapshot tensors")
            if type(entry.count) is not int or entry.count < 0:
                raise ValueError("Cache count must be a nonnegative integer")
            if type(entry.pointer) is not int:
                raise ValueError("Cache pointer must be an integer")
            capacity = entry.keys.shape[2]
            if entry.bounded:
                if capacity == 0 or entry.pointer != entry.count % capacity:
                    raise ValueError("Invalid bounded cache pointer or capacity")
            elif entry.pointer != 0 or capacity != entry.count:
                raise ValueError("Invalid unbounded cache count or pointer")
            entries.append(entry)

        return cls(
            entries=entries,
            storage_device=storage_device,
            read_device=read_device,
        )

    def step(self, layer_id: str) -> int:
        return self.entries[layer_id].count

    def write(
        self,
        layer_id: str,
        keys: Tensor,
        values: Tensor,
    ) -> None:
        entry = self.entries[layer_id]

        keys = keys.to(self.storage_device)
        values = values.to(self.storage_device)

        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("Cache tensors must have four dimensions")

        if keys.shape != values.shape:
            raise ValueError("Keys and values must have matching shapes")

        if keys.dtype != values.dtype:
            raise ValueError("Keys and values must have matching dtypes")

        # Definition-created entries acquire batch size and dtype on first write.
        if entry.count == 0 and entry.keys.shape[0] == 0:
            if (keys.shape[1], keys.shape[3]) != (
                entry.keys.shape[1],
                entry.keys.shape[3],
            ):
                raise ValueError("Incoming tensor dimensions do not match the cache")
            shape = (keys.shape[0], keys.shape[1], entry.keys.shape[2], keys.shape[3])
            entry.keys = keys.new_empty(shape)
            entry.values = values.new_empty(shape)

        for incoming, stored in (
            (keys, entry.keys),
            (values, entry.values),
        ):
            if (
                incoming.shape[0] != stored.shape[0]
                or incoming.shape[1] != stored.shape[1]
                or incoming.shape[3] != stored.shape[3]
            ):
                raise ValueError("Incoming tensor dimensions do not match the cache")

            if incoming.dtype != stored.dtype:
                raise ValueError("Incoming tensor dtype does not match the cache")

            if incoming.device != stored.device:
                raise ValueError("Incoming tensor device does not match the cache")

        new_token_count = keys.shape[2]

        if not entry.bounded:
            entry.keys = torch.cat((entry.keys, keys), dim=2)
            entry.values = torch.cat((entry.values, values), dim=2)
            entry.count += new_token_count
            return

        capacity = entry.keys.shape[2]

        if capacity == 0:
            raise ValueError("Bounded cache capacity must be greater than zero")

        skipped = max(0, new_token_count - capacity)
        retained = new_token_count - skipped
        start = (entry.pointer + skipped) % capacity
        first_length = min(retained, capacity - start)
        second_length = retained - first_length

        for incoming, stored in ((keys, entry.keys), (values, entry.values)):
            stored[:, :, start : start + first_length, :] = incoming[
                :, :, skipped : skipped + first_length, :
            ]
            if second_length:
                stored[:, :, :second_length, :] = incoming[
                    :, :, skipped + first_length :, :
                ]

        entry.pointer = (entry.pointer + new_token_count) % capacity
        entry.count += new_token_count

    def read_all(self, layer_id: str) -> CacheEntry:
        entry = self.entries[layer_id]

        if not entry.bounded:
            keys = entry.keys
            values = entry.values
        else:
            capacity = entry.keys.shape[2]

            if entry.count < capacity:
                keys = entry.keys[:, :, : entry.count, :]
                values = entry.values[:, :, : entry.count, :]
            else:
                keys = torch.cat(
                    (
                        entry.keys[:, :, entry.pointer :, :],
                        entry.keys[:, :, : entry.pointer, :],
                    ),
                    dim=2,
                )

                values = torch.cat(
                    (
                        entry.values[:, :, entry.pointer :, :],
                        entry.values[:, :, : entry.pointer, :],
                    ),
                    dim=2,
                )

        return CacheEntry(
            layer_id=entry.layer_id,
            bounded=entry.bounded,
            keys=keys.to(self.read_device),
            values=values.to(self.read_device),
            count=entry.count,
            pointer=0,
        )
