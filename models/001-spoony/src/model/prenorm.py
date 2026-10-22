"""Pre-normalize a branch; build_block owns the outer residual connection."""

from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.composition import Identifiable


class PreNormImpl(nn.Module):
    def __init__(self, norm: nn.Module, branch: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.branch = branch

    def assign_id(self, layer_id: int) -> None:
        if isinstance(self.branch, Identifiable):
            self.branch.assign_id(layer_id)

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        return self.branch(self.norm(x, cache=cache), cache=cache)
