"""Runtime composition and construction shared by block definitions."""

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from src.model.definitions import BlockDefinition


class Residual(nn.Module):
    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, x: Tensor) -> Tensor:
        output = self.branch(x)
        if output.shape != x.shape:
            raise ValueError("Residual branch output shape must match its input")
        return x + output


def build_block(definition: "BlockDefinition") -> nn.Module:
    """Build a fresh branch and apply its declared residual exactly once."""
    if type(definition.residual) is not bool:
        raise ValueError("residual must be a boolean")
    if definition.residual and definition.input_features != definition.output_features:
        raise ValueError("Residual requires equal input and output features")
    branch = definition.build()
    if not isinstance(branch, nn.Module):
        raise TypeError("Block build() must return an nn.Module")
    return Residual(branch) if definition.residual else branch
