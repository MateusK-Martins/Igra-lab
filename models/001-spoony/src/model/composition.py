"""Runtime composition and construction shared by block definitions."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor, nn

from src.model.cache_storage import CacheStorage

if TYPE_CHECKING:
    from src.model.definitions import BlockDefinition


@runtime_checkable
class Identifiable(Protocol):
    def assign_id(self, layer_id: int) -> None: ...


class ResidualImpl(nn.Module):
    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        output = self.branch(x, cache=cache)
        if output.shape != x.shape:
            raise ValueError("ResidualImpl branch output shape must match its input")
        return x + output


def build_block(
    definition: "BlockDefinition", layer_id: int | None = None
) -> nn.Module:
    """Build a fresh branch and apply its declared residual exactly once."""
    if type(definition.residual) is not bool:
        raise ValueError("residual must be a boolean")
    if definition.residual and definition.input_features != definition.output_features:
        raise ValueError("ResidualImpl requires equal input and output features")

    branch = definition.build()
    if not isinstance(branch, nn.Module):
        raise TypeError("Block build() must return an nn.Module")
    if layer_id is not None and isinstance(branch, Identifiable):
        branch.assign_id(layer_id)
    return ResidualImpl(branch) if definition.residual else branch
