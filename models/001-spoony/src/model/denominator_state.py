"""No-K signed state with learned-gain historical-weight normalization."""

import torch
from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.decay import TokenDecayImpl
from src.model.scan import affine_prefix_sum
from src.model.state_mechanisms import SignedStateImpl


class DenominatorStateImpl(SignedStateImpl):
    def __init__(
        self,
        features: int,
        eps: float,
        decay_init: float,
        mode: str,
        write_gate: bool = False,
    ):
        super().__init__(features, eps, decay_init, token_decay=True)
        self.eps, self.mode = eps, mode
        # Retain exactly the original per-feature learned gain, initialized to one.
        self.gain = nn.Parameter(self.value_norm.weights.detach().clone())
        del self.value_norm
        self.write_gate = None
        if write_gate:
            with torch.random.fork_rng(devices=[]):
                self.write_gate = TokenDecayImpl(features, 0.5)

    def assign_id(self, layer_id: int):
        suffix = "-write" if self.write_gate is not None else ""
        self.layer_id = f"DenominatorStateImpl-{self.mode}{suffix}-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None):
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Denominator-state cache requires no_grad")
        q, u = self.q_proj(x).sigmoid(), self.v_proj(x).tanh()
        dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
        u = u.to(dtype)
        if self.write_gate is not None:
            u = self.write_gate(x).to(dtype) * u
        gate = self.value_decay_gate(x).to(dtype)
        initial = None
        if cache is not None:
            previous = cache.read_state(self.layer_id)
            if previous.count:
                expected = (x.shape[0], 1, self.features)
                if (
                    previous.key_sum.shape != expected
                    or previous.value_sum.shape != expected
                    or previous.key_sum.dtype != dtype
                    or previous.value_sum.dtype != dtype
                ):
                    raise ValueError("Denominator cache dimensions or dtype changed")
                initial = torch.cat((previous.value_sum, previous.key_sum), -1)
        denominator_gate = gate if self.mode == "average" else gate.square()
        packed = affine_prefix_sum(
            torch.cat((u, torch.ones_like(u)), -1),
            torch.cat((gate, denominator_gate), -1),
            initial,
        )
        numerator, denominator = packed.chunk(2, -1)
        divisor = denominator + self.eps
        if self.mode == "squared":
            divisor = divisor.sqrt()
        read = self.gain.to(dtype) * numerator / divisor
        if cache is not None and x.shape[1]:
            # Generic key_sum stores the denominator; no K projection/state exists.
            cache.write_state(
                self.layer_id, denominator[:, -1:], numerator[:, -1:], x.shape[1]
            )
        return q * read.to(q.dtype)
