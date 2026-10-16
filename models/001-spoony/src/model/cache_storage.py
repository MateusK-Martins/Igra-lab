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


@dataclass
class MatrixCacheEntry:
    layer_id: str
    matrices: Tensor
    count: int = 0
    keys: Tensor | None = None


class CacheStorage:
    def __init__(
        self,
        entries: list[CacheEntry | MatrixCacheEntry],
        storage_device: torch.device,
        read_device: torch.device,
    ) -> None:
        self.storage_device = storage_device
        self.read_device = read_device

        self.entries: dict[str, CacheEntry] = {}
        self.matrices: dict[str, MatrixCacheEntry] = {}

        for entry in entries:
            if entry.layer_id in self.entries or entry.layer_id in self.matrices:
                raise ValueError(f"Duplicate cache layer_id: {entry.layer_id!r}")

            if isinstance(entry, MatrixCacheEntry):
                if (
                    entry.matrices.ndim != 4
                    or entry.matrices.shape[-2:] != (4, 4)
                    or type(entry.count) is not int
                    or entry.count < 0
                    or entry.matrices.shape[1] != entry.count
                    or not entry.matrices.is_floating_point()
                ):
                    raise ValueError("Invalid matrix cache state")
                if entry.keys is not None:
                    if (
                        entry.keys.shape != (*entry.matrices.shape[:2], 4)
                        or entry.keys.dtype != entry.matrices.dtype
                    ):
                        raise ValueError("Invalid matrix cache keys")
                    entry.keys = entry.keys.to(self.storage_device)
                entry.matrices = entry.matrices.to(self.storage_device)
                self.matrices[entry.layer_id] = entry
                continue
            entry.keys = entry.keys.to(self.storage_device)
            entry.values = entry.values.to(self.storage_device)

            self.entries[entry.layer_id] = entry

    def save(self, path: Path) -> None:
        snapshot = {
            "format_version": (
                3 if any(e.keys is not None for e in self.matrices.values()) else 2
            )
            if self.matrices
            else 1,
            "matrices": [
                {
                    "layer_id": e.layer_id,
                    "matrices": e.matrices.detach().cpu(),
                    "count": e.count,
                    **({"keys": e.keys.detach().cpu()} if e.keys is not None else {}),
                }
                for e in self.matrices.values()
            ],
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

        if not self.matrices:
            snapshot.pop("matrices")
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
        if not isinstance(snapshot, dict) or snapshot.get("format_version") not in (
            1,
            2,
            3,
        ):
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

        if snapshot["format_version"] in (2, 3):
            records = snapshot.get("matrices")
            if not isinstance(records, list):
                raise ValueError("Matrix snapshot records must be a list")
            for record in records:
                fields = {"layer_id", "matrices", "count"}
                if (
                    snapshot["format_version"] == 3
                    and isinstance(record, dict)
                    and "keys" in record
                ):
                    fields.add("keys")
                if not isinstance(record, dict) or set(record) != fields:
                    raise ValueError("Invalid matrix snapshot fields")
                if not isinstance(record["layer_id"], str) or not record["layer_id"]:
                    raise ValueError("Invalid matrix layer ID")
                if not isinstance(record["matrices"], Tensor):
                    raise TypeError("Invalid matrix tensor")
                if "keys" in record and not isinstance(record["keys"], Tensor):
                    raise TypeError("Invalid matrix key tensor")
                entries.append(MatrixCacheEntry(**record))
        return cls(
            entries=entries,
            storage_device=storage_device,
            read_device=read_device,
        )

    def step(self, layer_id: str) -> int:
        if layer_id in self.matrices:
            return self.matrices[layer_id].count
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

    def write_matrices(
        self, layer_id: str, matrices: Tensor, *, keys: Tensor | None = None
    ) -> None:
        entry = self.matrices[layer_id]
        if matrices.ndim != 4 or matrices.shape[-2:] != (4, 4):
            raise ValueError("Matrix cache expects [batch, tokens, 4, 4]")
        if (keys is None) != (entry.keys is None):
            raise ValueError("Matrix cache key mode mismatch")
        if keys is not None and (
            keys.shape != (*matrices.shape[:2], 4) or keys.dtype != matrices.dtype
        ):
            raise ValueError("Matrix cache key shape or dtype mismatch")
        matrices = matrices.detach().to(self.storage_device)
        first_write = entry.count == 0 and entry.matrices.shape[0] == 0
        if not first_write and (
            matrices.shape[0] != entry.matrices.shape[0]
            or matrices.dtype != entry.matrices.dtype
        ):
            raise ValueError("Matrix cache batch or dtype mismatch")
        # Validate both tensors before changing either part of the cache.
        previous = (
            matrices.new_empty((matrices.shape[0], 0, 4, 4))
            if first_write
            else entry.matrices
        )
        next_matrices = torch.cat((previous, matrices), dim=1)
        if keys is not None:
            keys = keys.detach().to(self.storage_device)
            previous_keys = (
                keys.new_empty((keys.shape[0], 0, 4)) if first_write else entry.keys
            )
            next_keys = torch.cat((previous_keys, keys), dim=1)
            entry.keys = next_keys
        entry.matrices = next_matrices
        entry.count += matrices.shape[1]

    def read_matrices(self, layer_id: str) -> Tensor:
        return self.matrices[layer_id].matrices.to(self.read_device)

    def read_matrix_keys(self, layer_id: str) -> Tensor:
        keys = self.matrices[layer_id].keys
        if keys is None:
            raise ValueError("Matrix cache has no independent keys")
        return keys.to(self.read_device)
