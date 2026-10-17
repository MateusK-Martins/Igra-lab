"""Diagonal-decay recurrent matrix fast weights."""

import math

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


class MatrixFastWeightsImpl(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        timescales: tuple[float, ...],
        learned_decay: bool,
    ) -> None:
        super().__init__()
        self.width = width
        self.timescales = timescales
        self.learned_decay = learned_decay
        self.matrix_features = 64
        self.scale_count = len(timescales)
        self.matrix_proj = nn.Linear(width, self.matrix_features, bias=True)
        self.query_proj = nn.Linear(width, 8, bias=True)
        self.out_proj = nn.Linear(8, width, bias=False)
        self.scale_proj = (
            nn.Linear(width, len(timescales), bias=True)
            if len(timescales) > 1
            else None
        )
        if learned_decay:
            initial = timescales[0]
            logit = math.log(initial / (1 - initial))
            self.decay_logit = nn.Parameter(torch.full((64,), logit))
            fixed = timescales[1:]
        else:
            self.register_parameter("decay_logit", None)
            fixed = timescales
        self.register_buffer(
            "fixed_decays", torch.tensor(fixed, dtype=torch.float32), persistent=True
        )
        self.layer_id = "MatrixFastWeightsImpl"

    def assign_id(self, layer_id: int) -> None:
        self.layer_id = f"MatrixFastWeightsImpl-{layer_id}"

    def _decays(self, *, device: torch.device) -> Tensor:
        learned = (
            self.decay_logit.sigmoid().unsqueeze(0)
            if self.decay_logit is not None
            else self.fixed_decays.new_empty((0, 64))
        )
        fixed = self.fixed_decays.to(device=device).unsqueeze(-1).expand(-1, 64)
        return torch.cat((learned.to(device=device), fixed), dim=0)

    @staticmethod
    def _scan(writes: Tensor, decay: Tensor, initial: Tensor) -> Tensor:
        """Parallel closed-form scan for a feature-wise constant decay."""
        # writes [B,T,N,F], decay [N,F], initial [B,1,N,F]. Accumulate
        # low-precision model activations in FP32, retaining float64 references.
        dtype = torch.float64 if writes.dtype == torch.float64 else torch.float32
        writes = writes.to(dtype)
        decay = decay.to(dtype)
        initial = initial.to(dtype)
        length = writes.shape[1]
        if length == 0:
            return writes
        exponent = torch.arange(1, length + 1, device=writes.device, dtype=dtype)
        powers = decay.unsqueeze(0).pow(exponent[:, None, None])
        increments = (1 - decay)[None, None] * writes
        scaled = increments / powers[None]
        prefix = torch.cumsum(scaled, dim=1)
        return powers[None] * (initial + prefix)

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError(
                "Cached matrix fast weights are inference-only; use no_grad"
            )
        raw = self.matrix_proj(x)
        q = self.query_proj(x)
        decay = self._decays(device=x.device)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        raw_acc = raw.to(dtype).unsqueeze(2).expand(-1, -1, self.scale_count, -1)
        if cache is None:
            initial = raw_acc.new_zeros((x.shape[0], 1, self.scale_count, 64))
        else:
            initial = cache.read_state(self.layer_id, like=raw_acc).reshape(
                x.shape[0], 1, self.scale_count, 64
            )
        states = self._scan(raw_acc, decay, initial)
        if cache is not None:
            cache.write_state(
                self.layer_id,
                states[:, -1:, :, :].reshape(x.shape[0], 1, -1),
                tokens=x.shape[1],
            )

        if self.scale_count > 1:
            assert self.scale_proj is not None
            scale_weights = torch.softmax(self.scale_proj(x).to(dtype), dim=-1)
            with torch.autocast(device_type=x.device.type, enabled=False):
                selected = torch.einsum(
                    "btn,btnf->btf", scale_weights.to(dtype), states.to(dtype)
                )
        else:
            selected = states[:, :, 0, :]
        matrices = selected.reshape(*selected.shape[:2], 8, 8)
        with torch.autocast(device_type=x.device.type, enabled=False):
            response = torch.einsum("btij,btj->bti", matrices, q.to(dtype))
        return self.out_proj(response.to(x.dtype))
