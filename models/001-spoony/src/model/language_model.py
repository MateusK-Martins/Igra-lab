from torch import Tensor, nn

from src.model.cache_storage import CacheStorage
from src.model.composition import ResidualImpl
from src.model.token_mlp import TokenMLPImpl


class ModelImpl(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        body: nn.ModuleList,
        lm_head: nn.Module,
        token_mlp_projections: nn.ModuleList | None = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.body = body
        self.lm_head = lm_head
        self.token_mlp_projections = token_mlp_projections

    def forward(
        self, token_ids: Tensor, *, cache: CacheStorage | None = None
    ) -> Tensor:
        hidden = self.embedding(token_ids)

        for block in self.body:
            branch = block.branch if isinstance(block, ResidualImpl) else block
            if isinstance(branch, TokenMLPImpl):
                if self.token_mlp_projections is None:
                    raise ValueError("TokenMLP requires shared down/up projections")
                hidden = block(
                    hidden,
                    cache=cache,
                    token_projections=tuple(self.token_mlp_projections),
                )
            else:
                hidden = block(hidden, cache=cache)

        return self.lm_head(hidden)
