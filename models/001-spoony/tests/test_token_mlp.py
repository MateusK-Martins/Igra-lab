from types import SimpleNamespace

import pytest
import torch
from torch import nn

from configs.model.model import build_model
from src.model.assembler import Model
from src.model.definitions import Embedding, LMHead, TokenMLP
from src.model.token_mlp import TokenMLPImpl


def test_chunked_output_and_gradients_match_elementwise_sigmoid_gates():
    torch.manual_seed(13)
    inputs = [
        torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]
    x, a, b = inputs
    expected = x * torch.sigmoid(a) * torch.sigmoid(b)
    actual = TokenMLPImpl(chunk_size=2)(
        x, token_vectors=(a, b), token_projections=(nn.Identity(), nn.Identity())
    )
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, inputs, upstream, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_shared_token_gates_looked_up_once_and_accumulate_layer_gradients():
    model, _ = Model(
        tokenizer=SimpleNamespace(vocab_size=7),
        input=Embedding(3, init_std=0.02),
        blocks=[TokenMLP(3, bottleneck_features=2, chunk_size=2, residual=True)] * 6,
        lm_head=LMHead(3, tie_to_embedding=True),
    ).assemble()
    calls = [0, 0]

    def count(index):
        def hook(module, args, output):
            calls[index] += 1

        return hook

    for index, table in enumerate(model.token_mlp_embeddings):
        table.register_forward_hook(count(index))
    ids = torch.tensor([[1, 2, 1]])
    actual = model(ids)
    assert calls == [1, 1]
    a, b = [torch.sigmoid(table.weight[ids]) for table in model.token_mlp_embeddings]
    hidden = model.embedding(ids)
    down, up = model.token_mlp_projections
    for _ in range(6):
        projected = down(hidden)
        hidden = hidden + up(projected * a * b)
    expected = model.lm_head(hidden)
    torch.testing.assert_close(actual, expected)
    parameters = list(model.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), parameters)
    actual.sum().backward()
    for parameter, expected_grad in zip(parameters, expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected_grad)
    assert len(parameters) == 5
    assert all(not list(block.parameters()) for block in model.body)
    for table in model.token_mlp_embeddings:
        assert torch.count_nonzero(table.weight.grad[[0, 3, 4, 5, 6]]) == 0
        assert torch.count_nonzero(table.weight.grad[[1, 2]]) > 0


def test_parameter_budget():
    model, _ = build_model(SimpleNamespace(vocab_size=8192), variant="learned_pair")
    assert sum(p.numel() for p in model.parameters()) == 6_459_904
    branches = [
        block.branch
        for block in model.body
        if hasattr(block, "branch") and isinstance(block.branch, TokenMLPImpl)
    ]
    assert len(branches) == 6
    assert all(not list(branch.parameters()) for branch in branches)
    tables = model.token_mlp_embeddings
    assert len(tables) == 2
    assert all(table.embedding_dim == 64 for table in tables)
    assert len({table.weight.data_ptr() for table in tables}) == 2
    assert model.token_mlp_projections[0].weight.shape == (64, 384)
    assert model.token_mlp_projections[1].weight.shape == (384, 64)


def test_context_variant_generates_second_sigmoid_gate_from_hidden_state():
    model, _ = Model(
        tokenizer=SimpleNamespace(vocab_size=7),
        input=Embedding(3, init_std=0.02),
        blocks=[
            TokenMLP(
                3,
                bottleneck_features=2,
                chunk_size=2,
                gate_mode="context",
                residual=True,
            )
        ],
        lm_head=LMHead(3, tie_to_embedding=True),
    ).assemble()
    ids = torch.tensor([[1, 2]])
    actual = model(ids)
    initial = model.embedding(ids)
    learned_gate = torch.sigmoid(model.token_mlp_embeddings[0](ids))
    down, up = model.token_mlp_projections
    projected = down(initial)
    expected_hidden = initial + up(projected * learned_gate * torch.sigmoid(projected))
    torch.testing.assert_close(actual, model.lm_head(expected_hidden))
    assert len(model.token_mlp_embeddings) == 1
    assert sum(p.numel() for p in model.parameters()) == 7 * 3 + 7 * 2 + 3 * 2 + 2 * 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_features": 0},
        {"input_features": True},
        {"input_features": 4, "chunk_size": 0},
        {"input_features": 4, "chunk_size": True},
        {"input_features": 4, "init_std": float("nan")},
        {"input_features": 4, "init_std": 0},
        {"input_features": 4, "residual": 1},
    ],
)
def test_invalid_definitions(kwargs):
    with pytest.raises(ValueError):
        TokenMLP(bottleneck_features=2, **kwargs)


def test_shared_projections_require_matching_widths():
    with pytest.raises(ValueError, match="matching widths"):
        Model(
            tokenizer=SimpleNamespace(vocab_size=7),
            input=Embedding(3),
            blocks=[TokenMLP(3, 2), TokenMLP(3, 4)],
            lm_head=LMHead(3),
        ).assemble()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bfloat16_backward():
    model, _ = build_model(SimpleNamespace(vocab_size=300))
    model.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(torch.randint(300, (2, 16), device="cuda"))
        loss = logits.float().square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_invalid_bottleneck_width(width):
    with pytest.raises(ValueError, match="bottleneck_features"):
        TokenMLP(input_features=384, bottleneck_features=width)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_cuda_transform_matches_elementwise_output_and_gradients(dtype):
    torch.manual_seed(71)
    x = torch.randn(6, 5, device="cuda", dtype=dtype, requires_grad=True)
    a = (torch.randn(6, 5, device="cuda") * 1.5).requires_grad_()
    b = (torch.randn(6, 5, device="cuda") * 1.5).requires_grad_()
    expected = x.float() * torch.sigmoid(a.float()) * torch.sigmoid(b.float())
    actual = TokenMLPImpl.transform(x, a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (x, a, b), upstream)
    actual_grads = torch.autograd.grad(actual, (x, a, b), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0.01, atol=1e-5)


def test_shared_tables_require_consistent_initialization_settings():
    with pytest.raises(ValueError, match="init_std"):
        Model(
            tokenizer=SimpleNamespace(vocab_size=7),
            input=Embedding(3),
            blocks=[TokenMLP(3, 2, init_std=0.02), TokenMLP(3, 2, init_std=0.1)],
            lm_head=LMHead(3),
        ).assemble()
