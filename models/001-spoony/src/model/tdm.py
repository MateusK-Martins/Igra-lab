"""Causal adjacent-window temporal differences with learned scale selection."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage


class TemporalDifferenceMixerImpl(nn.Module):
    def __init__(self, features: int, scales: tuple[int, ...]):
        super().__init__()
        self.features = features
        self.scales = scales
        self.history_size = 2 * max(scales) - 1
        self.scale_weights = nn.Parameter(torch.empty(len(scales), features, features))
        for weight in self.scale_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.router = nn.Linear(features, len(scales), bias=False)
        nn.init.zeros_(self.router.weight)
        self.out_proj = nn.Linear(features, features, bias=False)
        self.layer_id = ""

    def assign_id(self, layer_id: int):
        self.layer_id = f"TemporalDifferenceMixerImpl-{layer_id}"

    def temporal_differences(self, x: Tensor) -> Tensor:
        """[B,T,D] -> [B,T,S,D]; missing past is zero, divisor is always m."""
        working = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
        batch, length, features = x.shape
        if not length or not batch:
            return working[:, :, None, :].expand(
                batch, length, len(self.scales), features
            )
        prefix = F.pad(working.cumsum(dim=1), (0, 0, 1, 0))
        ends = torch.arange(1, length + 1, device=x.device)
        scales = torch.tensor(self.scales, device=x.device)
        recent = (ends[None, :] - scales[:, None]).clamp_min(0)
        previous = (ends[None, :] - 2 * scales[:, None]).clamp_min(0)
        shape = (batch, len(self.scales), length, features)
        difference = (
            prefix[:, 1:, :][:, None, :, :]
            - 2 * prefix.index_select(1, recent.flatten()).reshape(shape)
            + prefix.index_select(1, previous.flatten()).reshape(shape)
        ) / scales[None, :, None, None]
        return difference.permute(0, 2, 1, 3)

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError("TDM expects [B,T,features]")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("TDM cached execution requires no_grad")
        if not x.shape[1] or not x.shape[0]:
            return x * 0
        history = x
        if cache is not None:
            previous = cache.read_history(self.layer_id)
            if (
                previous.capacity != self.history_size
                or previous.values.shape[-1] != self.features
            ):
                raise ValueError("TDM cache capacity or feature width changed")
            if previous.count:
                if (
                    previous.values.shape[0] != x.shape[0]
                    or previous.values.dtype != x.dtype
                ):
                    raise ValueError("TDM cache batch or dtype changed")
                history = torch.cat((previous.values, x), dim=1)
        differences = self.temporal_differences(history)[:, -x.shape[1] :]
        projected = torch.einsum("btsd,sod->btso", differences, self.scale_weights)
        logits = self.router(x)
        logits = (
            logits.float()
            if logits.dtype in (torch.bfloat16, torch.float16)
            else logits
        )
        alpha = logits.softmax(dim=-1)
        mixed = (alpha[..., None] * projected.to(differences.dtype)).sum(dim=2)
        output = self.out_proj(mixed)
        if cache is not None:
            cache.write_history(self.layer_id, x)
        return output
