from types import SimpleNamespace

import pytest
import torch

from src.model.assembler import Model
from src.model.composition import build_block
from src.model.definitions import Embedding, LMHead, RMSNorm, Sequential, SwiGLU


def test_rmsnorm_normalizes_each_token_and_applies_learned_weights() -> None:
    module = RMSNorm(2, eps=1e-6).build().double()
    torch.testing.assert_close(module.weights, torch.ones(2, dtype=torch.double))
    with torch.no_grad():
        module.weights.copy_(torch.tensor([2.0, 3.0]))
    x = torch.tensor(
        [[[3.0, 4.0], [30.0, 40.0]], [[1.0, 2.0], [0.0, 0.0]]], dtype=torch.double
    )
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + module.eps)
    torch.testing.assert_close(module(x), expected * module.weights)
    torch.testing.assert_close(module(x)[0, :1], module(x[0, :1]))


def test_rmsnorm_double_precision_gradients() -> None:
    module = RMSNorm(3, eps=1e-6).build().double()
    x = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    module(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert module.weights.grad is not None
    assert torch.isfinite(module.weights.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_dtype_and_large_values_remain_finite(dtype) -> None:
    module = RMSNorm(2, eps=1e-6).build().to(dtype=dtype)
    x = torch.tensor([[[1000.0, 2000.0], [0.0, 0.0]]], dtype=dtype)
    working = x.float()
    expected = (
        working * torch.rsqrt(working.square().mean(-1, keepdim=True) + module.eps)
    ).to(dtype)
    actual = module(x)
    assert actual.dtype == dtype
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("eps", [0, -1, float("inf"), float("nan"), True, "small"])
def test_rmsnorm_rejects_invalid_epsilon(eps) -> None:
    with pytest.raises(ValueError, match="eps"):
        RMSNorm(4, eps)


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_rmsnorm_rejects_invalid_width(width) -> None:
    with pytest.raises(ValueError, match="input_features"):
        RMSNorm(width, 1e-6)


def test_rmsnorm_residual_and_model_integration() -> None:
    with pytest.raises(ValueError, match="boolean"):
        RMSNorm(4, 1e-6, residual=1)
    module = build_block(RMSNorm(4, 1e-6, residual=True))
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(module(x), x + module.branch(x))
    model, entries = Model(
        SimpleNamespace(vocab_size=11),
        Embedding(4),
        [Sequential([RMSNorm(4, 1e-6), SwiGLU(4, 8, 4, residual=True)])],
        LMHead(4, True),
    ).assemble()
    assert entries == []
    logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 11)
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
