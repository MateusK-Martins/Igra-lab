"""Controlled signed-state write/readout ablations with unchanged vector scan."""

import math

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.decay import TokenDecayImpl
from src.model.scan import affine_prefix_sum
from src.model.stability import HeadRMSNormImpl
from src.model.state_mechanisms import SignedStateImpl


class NoKVariantImpl(SignedStateImpl):
    def __init__(
        self,
        features: int,
        eps: float,
        decay_init: float,
        mode: str,
        rank: int,
        num_heads: int = 1,
        retention_initials: tuple[float, ...] | None = None,
        write_initials: tuple[float, ...] | None = None,
        normalization_scope: str = "head",
    ):
        super().__init__(features, eps, decay_init, token_decay=True)
        self.mode, self.rank = mode, rank
        self.write_gate = None
        self.read_down = self.read_up = None
        with torch.random.fork_rng(devices=[]):
            if mode == "write_gate":
                self.write_gate = TokenDecayImpl(features, 0.5)
            elif mode == "lowrank":
                self.read_down = nn.Linear(features, rank, bias=False)
                self.read_up = nn.Linear(rank, features, bias=False)
                nn.init.normal_(self.read_down.weight, std=1 / math.sqrt(features))
                nn.init.normal_(self.read_up.weight, std=1 / math.sqrt(rank))

        self.num_heads = num_heads
        self.normalization_scope = normalization_scope
        if num_heads > 1 and normalization_scope == "head":
            self.value_norm = HeadRMSNormImpl(features, eps, num_heads)
        for gate, values in (
            (self.value_decay_gate, retention_initials),
            (self.write_gate, write_initials),
        ):
            if values is not None:
                with torch.no_grad():
                    probabilities = gate.bias.new_tensor(values).repeat_interleave(
                        features // num_heads
                    )
                    gate.bias.copy_(torch.logit(probabilities))

    def assign_id(self, layer_id: int):
        suffix = f"-heads{self.num_heads}" if self.num_heads > 1 else ""
        if self.num_heads > 1 and self.normalization_scope == "global":
            suffix += "-global"
        self.layer_id = f"NoKVariantImpl-{self.mode}{suffix}-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("No-K variant cache requires no_grad")
        logits = self.q_proj(x)
        q = logits.tanh() if self.mode == "signed_readout" else logits.sigmoid()
        u = self.v_proj(x).tanh()
        dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
        u = u.to(dtype)
        if self.write_gate is not None:
            u = self.write_gate(x).to(dtype) * u
        initial = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                if (
                    previous.value_sum.shape != (x.shape[0], 1, self.features)
                    or previous.key_sum.shape != (x.shape[0], 1, 0)
                    or previous.value_sum.dtype != dtype
                ):
                    raise ValueError("No-K variant cache dimensions or dtype changed")
                initial = previous.value_sum
        state = affine_prefix_sum(u, self.value_decay_gate(x).to(dtype), initial)
        read = self.value_norm(state)
        if self.read_down is not None:
            read = self.read_up(self.read_down(read.to(self.read_down.weight.dtype)))
        if cache is not None and x.shape[1]:
            cache.write_state(
                self.layer_id,
                state.new_empty(x.shape[0], 1, 0),
                state[:, -1:],
                x.shape[1],
            )
        return q * read.to(q.dtype)
