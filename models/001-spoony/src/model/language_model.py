from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


class ModelImpl(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        body: nn.ModuleList,
        lm_head: nn.Module,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.body = body
        self.lm_head = lm_head

    def forward(
        self, token_ids: Tensor, *, cache: CacheStorage | None = None
    ) -> Tensor:
        hidden = self.embedding(token_ids)

        for block in self.body:
            hidden = block(hidden, cache=cache)

        return self.lm_head(hidden)
