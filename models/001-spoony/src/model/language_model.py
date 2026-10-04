from torch import Tensor, nn


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        body: nn.Module,
        lm_head: nn.Module,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.body = body
        self.lm_head = lm_head

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embedding(token_ids)
        hidden = self.body(hidden)
        return self.lm_head(hidden)
