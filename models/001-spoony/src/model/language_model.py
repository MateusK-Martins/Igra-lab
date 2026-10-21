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
        token_mlp_embeddings: nn.ModuleList | None = None,
        token_mlp_projections: nn.ModuleList | None = None,
        token_mlp_context_norm: nn.Module | None = None,
        token_mlp_context_vector_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.body = body
        self.lm_head = lm_head
        self.token_mlp_embeddings = token_mlp_embeddings
        self.token_mlp_projections = token_mlp_projections
        self.token_mlp_context_norm = token_mlp_context_norm
        self.token_mlp_context_vector_scale = token_mlp_context_vector_scale

    def forward(
        self, token_ids: Tensor, *, cache: CacheStorage | None = None
    ) -> Tensor:
        hidden = self.embedding(token_ids)

        token_vectors = None
        if self.token_mlp_embeddings is not None:
            (a,) = self.token_mlp_embeddings
            token_vectors = (a(token_ids),)

        for block in self.body:
            branch = block.branch if isinstance(block, ResidualImpl) else block
            if isinstance(branch, TokenMLPImpl):
                if (
                    token_vectors is None
                    or self.token_mlp_projections is None
                    or self.token_mlp_context_norm is None
                ):
                    raise ValueError(
                        "TokenMLP requires shared tables, projections and context norm"
                    )
                hidden = block(
                    hidden,
                    cache=cache,
                    token_vectors=token_vectors,
                    token_projections=tuple(self.token_mlp_projections),
                    context_norm=self.token_mlp_context_norm,
                    context_vector_scale=self.token_mlp_context_vector_scale,
                )
            else:
                hidden = block(hidden, cache=cache)

        return self.lm_head(hidden)
