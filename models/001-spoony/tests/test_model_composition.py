from dataclasses import dataclass

import pytest
import torch
from torch import nn

from src.model.composition import Residual, build_block
from src.model.definitions import Linear, Repeat, Sequential


@dataclass(frozen=True)
class Scale:
    factor: float
    input_features: int = 2
    output_features: int = 2
    residual: bool = False

    def build(self):
        layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2) * self.factor)
        return layer


def test_child_and_repeat_residual_scopes_and_gradients() -> None:
    # Each child maps x -> x + 2x = 3x; two repetitions produce 9x.
    # The outer residual adds the original x once, producing 10x.
    module = build_block(Repeat(2, [Scale(2, residual=True)], residual=True))
    x = torch.ones(1, 3, 2, requires_grad=True)
    torch.testing.assert_close(module(x), x * 10)
    module(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 10))
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_repeat_preserves_order_and_constructs_independent_parameters() -> None:
    module = build_block(Repeat(2, [Linear(2, 4), Linear(4, 2)]))
    assert isinstance(module, nn.Sequential)
    assert [(child.in_features, child.out_features) for child in module] == [
        (2, 4),
        (4, 2),
        (2, 4),
        (4, 2),
    ]
    assert module[0].weight is not module[2].weight
    assert module[0].weight.data_ptr() != module[2].weight.data_ptr()
    assert len(module.state_dict()) == 8
    assert module(torch.ones(1, 3, 2)).shape == (1, 3, 2)


def test_nested_repeat_keeps_outer_residual_scope() -> None:
    module = build_block(Repeat(2, [Repeat(2, [Scale(2)], residual=True)]))
    # Inner sequence is 4x; its residual gives 5x. Twice gives 25x.
    torch.testing.assert_close(module(torch.ones(1, 2)), torch.full((1, 2), 25.0))


def test_residual_rejects_broadcastable_wrong_shape() -> None:
    module = Residual(nn.Linear(2, 1))
    with pytest.raises(ValueError, match="shape"):
        module(torch.ones(1, 2))


def test_build_helper_rejects_incompatible_residual() -> None:
    with pytest.raises(ValueError, match="equal"):
        build_block(Linear(2, 4, residual=True))


def test_sequential_child_and_outer_residual_scopes() -> None:
    module = build_block(Sequential([Scale(2, residual=True), Scale(4)], residual=True))
    # Child residual produces 3x, next child gives 12x, outer adds x -> 13x.
    torch.testing.assert_close(module(torch.ones(1, 2)), torch.full((1, 2), 13.0))


def test_build_helper_rejects_non_module_return_and_bad_flag() -> None:
    from types import SimpleNamespace

    with pytest.raises(TypeError, match="nn.Module"):
        build_block(SimpleNamespace(residual=False, build=list))
    with pytest.raises(ValueError, match="boolean"):
        build_block(SimpleNamespace(residual=1))
