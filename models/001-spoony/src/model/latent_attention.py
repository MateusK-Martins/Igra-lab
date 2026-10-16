"""Single-head compressed attention and query-conditioned matrix attention."""

import torch
from torch import nn
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.position import rope


class LatentAttentionImpl(nn.Module):
    def __init__(self, width: int, mode: str, *, rope_base: float | None = None):
        super().__init__()
        self.rope_base = rope_base
        self.mode = mode
        self.latent_features = 16 if mode == "a16" else 4
        self.q_proj = nn.Linear(
            width,
            self.latent_features,
            bias=mode in ("sma4", "sma4_key", "sma4_key_gelu16"),
        )
        if mode in ("sma4", "sma4_key", "sma4_key_gelu16"):
            self.matrix_proj = nn.Linear(width, 16, bias=True)
            if mode == "sma4":
                self.scoring = nn.Parameter(torch.empty(4))
                nn.init.normal_(self.scoring, std=0.5)
        else:
            self.k_proj = nn.Linear(width, self.latent_features, bias=False)
            self.v_proj = nn.Linear(width, self.latent_features, bias=False)
        self.out_proj = nn.Linear(
            16 if mode == "sma4_key_gelu16" else self.latent_features,
            width,
            bias=mode in ("sma4", "sma4_key", "sma4_key_gelu16"),
        )

        if mode in ("sma4_key", "sma4_key_gelu16"):
            self.k_proj = nn.Linear(width, 4, bias=False)

        if mode == "sma4_key_gelu16":
            self.response_proj = nn.Linear(4, 16, bias=True)

    def assign_id(self, layer_id):
        self.layer_id = f"LatentAttentionImpl{'-rope' if self.rope_base is not None else ''}-{layer_id}"

    def forward(self, x, *, cache: CacheStorage | None = None):
        return self.out_proj(self.retrieve(x, cache=cache))

    def retrieve(self, x, *, cache: CacheStorage | None = None):
        """Return only the compressed retrieval; do not retain it between calls."""
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Cached attention is inference-only; use no_grad")
        start = 0 if cache is None else cache.step(self.layer_id)
        q = self.q_proj(x)
        positions = start + torch.arange(x.shape[1], device=x.device)
        if self.rope_base is not None:
            q = rope(q.unsqueeze(1), positions, base=self.rope_base).squeeze(1)
        if self.mode in ("sma4", "sma4_key", "sma4_key_gelu16"):
            matrices = self.matrix_proj(x).unflatten(-1, (4, 4))
            keys = (
                self.k_proj(x) if self.mode in ("sma4_key", "sma4_key_gelu16") else None
            )
            if self.rope_base is not None:
                # Rotate matrix rows as vectors of column coordinates: M_i R_i^T.
                matrices = rope(
                    matrices.transpose(1, 2), positions, base=self.rope_base
                ).transpose(1, 2)
                if keys is not None:
                    keys = rope(
                        keys.unsqueeze(1), positions, base=self.rope_base
                    ).squeeze(1)
            if cache is not None:
                cache.write_matrices(self.layer_id, matrices, keys=keys)
                matrices = cache.read_matrices(self.layer_id)
                if keys is not None:
                    keys = cache.read_matrix_keys(self.layer_id)
            # Preserve float64 references, accumulate low-precision responses in FP32.
            dtype = (
                torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
            )
            # Autocast must not demote the explicitly FP32 contractions.
            with torch.autocast(device_type=x.device.type, enabled=False):
                responses = torch.einsum(
                    "bsij,btj->btsi", matrices.to(dtype), q.to(dtype)
                )
                if keys is not None:
                    scores = (
                        torch.einsum("btd,bsd->bts", q.to(dtype), keys.to(dtype)) / 2
                    )
                else:
                    scores = (
                        torch.einsum("btsi,i->bts", responses, self.scoring.to(dtype))
                        / 2
                    )
                mask = torch.arange(matrices.shape[1], device=x.device)[None, :] <= (
                    start + torch.arange(x.shape[1], device=x.device)[:, None]
                )
                weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
                if self.mode == "sma4_key_gelu16":
                    # Transform every matrix response BEFORE the token-weighted sum.
                    responses = F.gelu(
                        F.linear(
                            responses,
                            self.response_proj.weight.to(dtype),
                            self.response_proj.bias.to(dtype),
                        )
                    )
                z = torch.einsum("bts,btsi->bti", weights, responses)
            return z.to(q.dtype)
        q = q.unsqueeze(1)
        k, v = self.k_proj(x).unsqueeze(1), self.v_proj(x).unsqueeze(1)
        if self.rope_base is not None:
            k = rope(k, positions, base=self.rope_base)
        if cache is not None:
            cache.write(self.layer_id, k, v)
            entry = cache.read_all(self.layer_id)
            k, v = entry.keys, entry.values
        mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (
            start + torch.arange(x.shape[1], device=x.device)[:, None]
        )
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return z.squeeze(1)
