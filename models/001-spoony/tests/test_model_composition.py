from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from src.model.assembler import Model
from src.model.composition import ResidualImpl, build_block
from src.model.definitions import (
    BlockDefinition,
    Embedding,
    Linear,
    LMHead,
    Repeat,
    Sequential,
)
from src.model.feedforward import LinearImpl


@dataclass(frozen=True)
class Scale(BlockDefinition):
    factor: float
    input_features: int = 2
    output_features: int = 2
    residual: bool = False

    def build(self):
        layer = LinearImpl(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2) * self.factor)
        return layer


def body_for(block):
    return (
        Model(SimpleNamespace(vocab_size=7), Embedding(2), [block], LMHead(2))
        .assemble()[0]
        .body
    )


def run(body, x):
    for module in body:
        x = module(x)
    return x


def test_repeated_child_residuals_and_gradients():
    body = body_for(Repeat(2, [Scale(2, residual=True)]))
    x = torch.ones(1, 3, 2, requires_grad=True)
    output = run(body, x)
    torch.testing.assert_close(output, x * 9)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 9))
    assert all(p.grad is not None for p in body.parameters())


def test_repeat_preserves_order_and_independent_parameters():
    body = body_for(Repeat(2, [Linear(2, 4), Linear(4, 2)]))
    assert [(m.in_features, m.out_features) for m in body] == [
        (2, 4),
        (4, 2),
        (2, 4),
        (4, 2),
    ]
    assert body[0].weight.data_ptr() != body[2].weight.data_ptr()
    assert len(body.state_dict()) == 8
    assert run(body, torch.ones(1, 3, 2)).shape == (1, 3, 2)


def test_nested_unpack_order_and_leaf_residuals():
    body = body_for(
        Sequential([Repeat(2, [Repeat(2, [Scale(2)])]), Scale(3, residual=True)])
    )
    assert len(body) == 5
    torch.testing.assert_close(run(body, torch.ones(1, 2)), torch.full((1, 2), 64.0))


@pytest.mark.parametrize(
    "group", [Repeat(2, [Linear(2, 2)]), Sequential([Linear(2, 2)])]
)
def test_compositions_must_be_unpacked_before_build(group):
    with pytest.raises(TypeError, match="unpacked"):
        group.build()


def test_residual_rejects_broadcastable_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        ResidualImpl(LinearImpl(2, 1))(torch.ones(1, 2))


def test_build_helper_validation():
    with pytest.raises(ValueError, match="equal"):
        build_block(Linear(2, 4, residual=True))
    with pytest.raises(TypeError, match="nn.Module"):
        build_block(SimpleNamespace(residual=False, build=list))
    with pytest.raises(ValueError, match="boolean"):
        build_block(SimpleNamespace(residual=1))
