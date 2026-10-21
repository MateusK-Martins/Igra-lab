from types import SimpleNamespace

import pytest
import torch
from torch import nn

from configs.model.model import build_model
from src.model.assembler import Model
from src.model.definitions import Embedding, LMHead, TokenMLP
from src.model.token_mlp import TokenMLPImpl


def test_projection_only_runtime_encodes_then_decodes_with_gradients():
    torch.manual_seed(13)
    x = torch.randn(2, 3, 5, requires_grad=True)
    down = nn.Linear(5, 2, bias=False)
    up = nn.Linear(2, 5, bias=False)
    actual = TokenMLPImpl()(x, token_projections=(down, up))
    expected = up(down(x))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in (*down.parameters(), *up.parameters())
    )


def test_shared_projections_accumulate_across_all_tokenmlp_layers():
    model, _ = Model(
        tokenizer=SimpleNamespace(vocab_size=7),
        input=Embedding(3, init_std=0.02),
        blocks=[TokenMLP(3, bottleneck_features=2, residual=True) for _ in range(6)],
        lm_head=LMHead(3, tie_to_embedding=True),
    ).assemble()
    ids = torch.tensor([[1, 2, 1]])
    actual = model(ids)
    hidden = model.embedding(ids)
    down, up = model.token_mlp_projections
    for _ in range(6):
        hidden = hidden + up(down(hidden))
    expected = model.lm_head(hidden)
    torch.testing.assert_close(actual, expected)
    parameters = list(model.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), parameters)
    actual.sum().backward()
    for parameter, expected_grad in zip(parameters, expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected_grad)
    assert not hasattr(model, "token_mlp_embeddings")
    assert model.token_mlp_projections[0].weight.shape == (2, 3)
    assert model.token_mlp_projections[1].weight.shape == (3, 2)


def test_projection_only_parameter_budget():
    model, _ = build_model(SimpleNamespace(vocab_size=8192))
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_411_328
    branches = [
        block.branch
        for block in model.body
        if hasattr(block, "branch") and isinstance(block.branch, TokenMLPImpl)
    ]
    assert len(branches) == 6
    assert all(not list(branch.parameters()) for branch in branches)
    assert model.token_mlp_projections[0].weight.shape == (64, 384)
    assert model.token_mlp_projections[1].weight.shape == (384, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"input_features": True},
        {"input_features": 4, "bottleneck_features": 0},
        {"input_features": 4, "bottleneck_features": True},
        {"input_features": 4, "residual": 1},
    ],
)
def test_invalid_definitions(kwargs):
    with pytest.raises(ValueError):
        TokenMLP(**{"bottleneck_features": 2, **kwargs})


def test_shared_projections_require_matching_widths():
    with pytest.raises(ValueError, match="matching widths"):
        Model(
            tokenizer=SimpleNamespace(vocab_size=7),
            input=Embedding(3),
            blocks=[TokenMLP(3, 2), TokenMLP(3, 4)],
            lm_head=LMHead(3),
        ).assemble()
