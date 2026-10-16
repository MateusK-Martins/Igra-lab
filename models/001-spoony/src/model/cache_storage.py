from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class StateEntry:
    """Recurrent state [B,1,D]; key_sum is empty in Tide (snapshot compatibility)."""

    layer_id: str
    key_sum: Tensor
    value_sum: Tensor
    count: int = 0


class CacheStorage:
    def __init__(
        self,
        entries: list[StateEntry],
        storage_device: torch.device,
        read_device: torch.device,
    ) -> None:
        self.storage_device = storage_device
        self.read_device = read_device

        self.states: dict[str, StateEntry] = {}

        for entry in entries:
            if not isinstance(entry, StateEntry):
                raise TypeError("CacheStorage requires recurrent StateEntry objects")
            if entry.layer_id in self.states:
                raise ValueError(f"Duplicate cache layer_id: {entry.layer_id!r}")
            entry.key_sum = entry.key_sum.to(self.storage_device)
            entry.value_sum = entry.value_sum.to(self.storage_device)
            self.states[entry.layer_id] = entry

    def save(self, path: Path) -> None:
        snapshot = {
            "format_version": 1,
            "states": [
                {
                    "layer_id": entry.layer_id,
                    "key_sum": entry.key_sum.detach().cpu(),
                    "value_sum": entry.value_sum.detach().cpu(),
                    "count": entry.count,
                }
                for entry in self.states.values()
            ],
            "entries": [],  # Retain the recurrent format-v1 snapshot schema.
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

        if records:
            raise ValueError("Attention KV cache snapshots are not supported by Tide")
        entries = []

        states = snapshot.get("states", [])
        if not isinstance(states, list):
            raise TypeError("Cache states must be a list")
        for record in states:
            if not isinstance(record, dict) or set(record) != {
                "layer_id",
                "key_sum",
                "value_sum",
                "count",
            }:
                raise ValueError("Invalid recurrent cache fields")
            entry = StateEntry(**record)
            if not isinstance(entry.layer_id, str) or not entry.layer_id:
                raise ValueError("Invalid recurrent cache layer_id")
            if type(entry.count) is not int or entry.count < 0:
                raise ValueError("Invalid recurrent cache count")
            if (
                not isinstance(entry.key_sum, Tensor)
                or not isinstance(entry.value_sum, Tensor)
                or entry.key_sum.ndim != 3
                or entry.value_sum.ndim != 3
                or entry.key_sum.shape[:2] != entry.value_sum.shape[:2]
                or entry.key_sum.shape[1] != 1
                or entry.value_sum.shape[2] == 0
                or entry.key_sum.dtype != entry.value_sum.dtype
                or not entry.key_sum.is_floating_point()
                or (entry.count > 0 and entry.key_sum.shape[0] == 0)
            ):
                raise ValueError("Invalid recurrent cache tensors")
            entries.append(entry)

        return cls(
            entries=entries,
            storage_device=storage_device,
            read_device=read_device,
        )

    def step(self, layer_id: str) -> int:
        return self.states[layer_id].count

    def read_state(self, layer_id: str) -> StateEntry:
        entry = self.states[layer_id]
        return StateEntry(
            entry.layer_id,
            entry.key_sum.to(self.read_device),
            entry.value_sum.to(self.read_device),
            entry.count,
        )

    def write_state(
        self, layer_id: str, key_sum: Tensor, value_sum: Tensor, tokens: int
    ) -> None:
        entry = self.states[layer_id]
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("State update requires a positive token count")
        if (
            key_sum.ndim != 3
            or value_sum.ndim != 3
            or key_sum.shape[:2] != value_sum.shape[:2]
            or key_sum.shape[1] != 1
            or key_sum.shape[2] != entry.key_sum.shape[2]
            or value_sum.shape[2] != entry.value_sum.shape[2]
            or key_sum.shape[0] == 0
            or key_sum.dtype != value_sum.dtype
            or not key_sum.is_floating_point()
        ):
            raise ValueError("Invalid recurrent state update")
        if entry.count and (
            key_sum.shape != entry.key_sum.shape
            or value_sum.shape != entry.value_sum.shape
            or key_sum.dtype != entry.key_sum.dtype
        ):
            raise ValueError("Recurrent state shape or dtype changed")
        keys = key_sum.detach().to(self.storage_device).clone()
        values = value_sum.detach().to(self.storage_device).clone()
        entry.key_sum, entry.value_sum = keys, values
        entry.count += tokens
