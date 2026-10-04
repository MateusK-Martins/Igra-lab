import pytest
import torch
from torch.nn import functional as F

from src.model.composition import build_block
from src.model.definitions import SwiGLU


@pytest.mark.parametrize("bias", [False, True])
def test_swiglu_formula_shapes_and_gradients(bias) -> None:
    module = SwiGLU(3, 5, 2, bias=bias).build().double()
    x = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    gate = F.linear(x, module.gate.weight, module.gate.bias)
    up = F.linear(x, module.up.weight, module.up.bias)
    expected = F.linear(F.silu(gate) * up, module.down.weight, module.down.bias)
    torch.testing.assert_close(module(x), expected)
    assert module(x).shape == (2, 4, 2)
    assert (module.gate.bias is not None) == bias
    assert torch.autograd.gradcheck(module, (x,))
    module(x).sum().backward()
    assert all(p.grad is not None for p in module.parameters())


@pytest.mark.parametrize(
    "field", ["input_features", "hidden_features", "output_features"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_swiglu_rejects_invalid_dimensions(field, value) -> None:
    settings = {"input_features": 4, "hidden_features": 8, "output_features": 4}
    settings[field] = value
    with pytest.raises(ValueError, match=field):
        SwiGLU(**settings)


@pytest.mark.parametrize("field", ["bias", "residual"])
def test_swiglu_rejects_non_boolean_flags(field) -> None:
    with pytest.raises(ValueError, match=field):
        SwiGLU(4, 8, 4, **{field: 1})


def test_swiglu_residual_is_applied_by_helper_only() -> None:
    recipe = SwiGLU(4, 8, 4, residual=True)
    module = build_block(recipe)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(module(x), x + module.branch(x))
