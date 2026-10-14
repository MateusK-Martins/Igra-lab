"""Headless associative matrix memory, using the existing affine scan in chunks."""

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.model.cache_storage import CacheStorage
from src.model.scan import affine_prefix_sum
from src.model.stability import RMSNormImpl


def matrix_reference(
    k: Tensor, v: Tensor, gate: Tensor, initial: Tensor | None = None
) -> Tensor:
    """Small independent sequential reference; inputs already include write scaling."""
    state = (
        k.new_zeros(k.shape[0], k.shape[-1], v.shape[-1])
        if initial is None
        else initial
    )
    states = []
    for t in range(k.shape[1]):
        state = gate[:, t] * state + k[:, t, :, None] * v[:, t, None, :]
        states.append(state)
    return (
        torch.stack(states, dim=1)
        if states
        else k.new_empty(k.shape[0], 0, k.shape[-1], v.shape[-1])
    )


class MatrixMemoryImpl(nn.Module):
    def __init__(
        self,
        features: int,
        key_features: int,
        value_features: int,
        eps: float,
        decay_init: float,
        chunk_size: int,
        checkpoint_chunks: bool = True,
    ):
        super().__init__()
        self.features, self.key_features, self.value_features = (
            features,
            key_features,
            value_features,
        )
        self.chunk_size, self.checkpoint_chunks = chunk_size, checkpoint_chunks
        self.q_proj = nn.Linear(features, features, bias=False)
        self.k_proj = nn.Linear(features, features, bias=False)
        self.v_proj = nn.Linear(features, features, bias=False)
        # Preserve the baseline RNG stream for unchanged FFNs and later layers.
        with torch.random.fork_rng(devices=[]):
            self.q_down = nn.Linear(features, key_features, bias=False)
            self.k_down = nn.Linear(features, key_features, bias=False)
            self.v_down = nn.Linear(features, value_features, bias=False)
            self.row_gate = nn.Linear(features, key_features)
            self.column_gate = nn.Linear(features, value_features)
            bias = math.log(math.sqrt(decay_init) / (1 - math.sqrt(decay_init)))
            for gate in (self.row_gate, self.column_gate):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, bias)
            self.read_norm = RMSNormImpl(value_features, eps)
            self.out_proj = nn.Linear(value_features, features, bias=False)
        self.layer_id = ""

    def assign_id(self, layer_id: int):
        self.layer_id = f"MatrixMemoryImpl-{layer_id}"

    def projections(self, x: Tensor):
        q = self.q_down(self.q_proj(x).sigmoid())
        k = self.k_down(self.k_proj(x).tanh())
        v = self.v_down(self.v_proj(x).sigmoid())
        # Gate logits/sigmoid and recurrent arithmetic remain FP32 under BF16 autocast.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.to(dtype)
            row = torch.nn.functional.linear(
                z, self.row_gate.weight.to(dtype), self.row_gate.bias.to(dtype)
            ).sigmoid()
            column = torch.nn.functional.linear(
                z, self.column_gate.weight.to(dtype), self.column_gate.bias.to(dtype)
            ).sigmoid()
        return (
            q.to(dtype),
            k.to(dtype) / math.sqrt(self.key_features),
            v.to(dtype),
            row,
            column,
        )

    def _chunk(self, q, k, v, row, column, initial):
        # Only this chunk materializes [B,C,K,V]; no sequence-sized matrix cache.
        update = k.unsqueeze(-1) * v.unsqueeze(-2)
        gate = row.unsqueeze(-1) * column.unsqueeze(-2)
        states = affine_prefix_sum(
            update.flatten(2), gate.flatten(2), initial.flatten(1).unsqueeze(1)
        )
        states = states.reshape_as(update)
        reads = (q.unsqueeze(-1) * states).sum(-2) / math.sqrt(self.key_features)
        return reads, states[:, -1]

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Matrix cached execution requires no_grad")
        q, k, v, row, column = self.projections(x)
        state = q.new_zeros(x.shape[0], self.key_features, self.value_features)
        if cache is not None:
            previous = cache.read_matrix_state(self.layer_id)
            if previous.count:
                if (
                    previous.matrix.shape != state.shape
                    or previous.matrix.dtype != state.dtype
                ):
                    raise ValueError("Matrix cache batch, dimensions or dtype changed")
                state = previous.matrix
        reads = []
        for start in range(0, x.shape[1], self.chunk_size):
            args = [
                z[:, start : start + self.chunk_size] for z in (q, k, v, row, column)
            ]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                read, state = checkpoint(self._chunk, *args, state, use_reentrant=False)
            else:
                read, state = self._chunk(*args, state)
            reads.append(read)
        if not reads:
            return x.new_empty(x.shape)
        result = self.read_norm(torch.cat(reads, dim=1))
        result = self.out_proj(result.to(self.out_proj.weight.dtype))
        if cache is not None:
            cache.write_matrix_state(self.layer_id, state, x.shape[1])
        return result
