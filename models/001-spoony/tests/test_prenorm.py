from types import SimpleNamespace

import pytest
import torch

from configs.model.model import build_model
from src.model.composition import ResidualImpl, build_block
from src.model.definitions import Linear, PreNorm
from src.model.prenorm import PreNormImpl


def test_prenorm_build_returns_unwrapped_branch_and_residual_preserves_raw_input():
    definition = PreNorm(Linear(4, 4))
    assert isinstance(definition.build(), PreNormImpl)
    runtime = build_block(definition)
    assert isinstance(runtime, ResidualImpl)
    runtime.branch.norm.weights.data.fill_(2)
    runtime.branch.branch.weight.data.zero_()
    runtime.branch.branch.bias.data.zero_()
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    torch.testing.assert_close(runtime(x), x, rtol=0, atol=0)


def test_prenorm_rejects_inner_residual_to_avoid_double_skip():
    with pytest.raises(ValueError, match="inner block"):
        PreNorm(Linear(4, 4, residual=True))


def test_model_uses_standard_llama_layer_equations():
    model, _ = build_model(SimpleNamespace(vocab_size=32))
    model.eval()
    x = torch.randn(1, 3, 512)
    attention, feedforward = model.body[:2]
    expected = x + attention.branch.branch(attention.branch.norm(x))
    expected = expected + feedforward.branch.branch(feedforward.branch.norm(expected))
    torch.testing.assert_close(feedforward(attention(x)), expected)
