from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.model.assembler import Model
from src.model.attention import GQAAttentionImpl
from src.model.cache_storage import CacheStorage
from src.model.definitions import (
    BlockDefinition,
    Embedding,
    Linear,
    LMHead,
    Repeat,
    RMSNorm,
    Sequential,
    SwiGLU,
)
from src.model.position import rope


class RecorderImpl(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, x, *, cache=None):
        self.received.append(cache)
        return x * 2


@dataclass(frozen=True)
class Recorder(BlockDefinition):
    input_features: int = 4
    output_features: int = 4
    residual: bool = False

    def build(self):
        return RecorderImpl()


@pytest.mark.parametrize("use_cache", [False, True])
def test_model_passes_same_cache_through_flattened_body_and_residuals(use_cache):
    model, entries = Model(
        SimpleNamespace(vocab_size=7),
        Embedding(4),
        [Repeat(2, [Sequential([Recorder(), Recorder(residual=True)])])],
        LMHead(4),
    ).assemble()
    cache = (
        CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
        if use_cache
        else None
    )
    tokens = torch.tensor([[1, 2]])
    output = model(tokens, cache=cache)
    expected = model.lm_head(model.embedding(tokens) * 36)
    torch.testing.assert_close(output, expected)
    recorders = [m for m in model.modules() if isinstance(m, RecorderImpl)]
    assert len(recorders) == 4
    assert all(len(m.received) == 1 and m.received[0] is cache for m in recorders)


@pytest.mark.parametrize("recipe", [Linear(4, 4), RMSNorm(4, 1e-6), SwiGLU(4, 8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stateless_blocks_ignore_cache_and_preserve_gradients(recipe, dtype):
    module = recipe.build().to(dtype=dtype)
    cache = CacheStorage([], torch.device("cpu"), torch.device("cpu"))
    x = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    plain = module(x)
    cached = module(x, cache=cache)
    torch.testing.assert_close(cached, plain)
    plain_grad = torch.autograd.grad(plain.square().sum(), x)[0]
    cached_grad = torch.autograd.grad(cached.square().sum(), x)[0]
    torch.testing.assert_close(cached_grad, plain_grad)
    assert cache.entries == {}


@pytest.mark.parametrize("residual", [False, True])
def test_attention_receives_storage_through_model(residual):
    from src.model.definitions import GQAAttention

    model, entries = Model(
        SimpleNamespace(vocab_size=7),
        Embedding(4),
        [
            GQAAttention(
                4,
                4,
                2,
                1,
                2,
                0,
                0.0,
                position_rotation=partial(rope, base=10000.0),
                residual=residual,
            )
        ],
        LMHead(4),
    ).assemble()
    cache = CacheStorage(entries, torch.device("cpu"), torch.device("cpu"))
    received = []

    def forward(module, x, *, cache=None):
        received.append((module.layer_id, cache))
        return x * 2

    tokens = torch.tensor([[1, 2]])
    with patch.object(GQAAttentionImpl, "forward", forward):
        output = model(tokens, cache=cache)
    assert received == [(entries[0].layer_id, cache)]
    torch.testing.assert_close(
        output, model.lm_head(model.embedding(tokens) * (3 if residual else 2))
    )
