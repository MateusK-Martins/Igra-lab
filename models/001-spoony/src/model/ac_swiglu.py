"""A16 attention residual followed by retrieval-conditioned SwiGLU."""

import torch
from torch import nn
from torch.nn import functional as F

from src.model.feedforward import SwiGLUImpl
from src.model.latent_attention import LatentAttentionImpl
from src.model.stability import RMSNormImpl


class ACSwiGLUImpl(nn.Module):
    def __init__(self, features, hidden_features, eps, rope_base, alpha, conditioned):
        super().__init__()
        self.attention_norm = RMSNormImpl(features, eps)
        # Preserve the original LatentAttention definition's initialization stream.
        with torch.random.fork_rng(devices=[]):
            self.attention = LatentAttentionImpl(features, "a16", rope_base=rope_base)
        self.ffn_norm = RMSNormImpl(features, eps)
        self.ffn = SwiGLUImpl(features, hidden_features, features, bias=False)
        self.alpha = alpha
        self.condition = None
        if conditioned:
            # A zero-initialized addition must not shift later shared weights.
            with torch.random.fork_rng(devices=[]):
                self.condition = nn.Linear(16, hidden_features, bias=False)
            nn.init.zeros_(self.condition.weight)

    def assign_id(self, layer_id):
        self.attention.assign_id(layer_id)
        self.layer_id = self.attention.layer_id

    def modulation(self, z):
        logits = self.condition(z)
        working = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        return 1 + self.alpha * working.tanh()

    def forward(self, x, *, cache=None):
        z = self.attention.retrieve(self.attention_norm(x), cache=cache)
        r = x + self.attention.out_proj(z)
        u = self.ffn_norm(r)
        h = F.silu(self.ffn.gate(u)) * self.ffn.up(u)
        if self.condition is not None:
            gamma = self.modulation(z)
            h = h * gamma
        return r + self.ffn.down(h)
