"""Turn a stream of token IDs into next-token prediction examples."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create fixed-length, shifted next-token prediction windows.

    The dataset is tokenizer-agnostic. It receives a one-dimensional tensor of
    token IDs and returns `(input_ids, target_ids)` pairs, each with shape
    `[context_length]`. Targets are the input sequence shifted by one token.
    """

    def __init__(
        self,
        token_ids: Tensor,
        context_length: int,
        stride: int,
    ) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional tensor")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long dtype")
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")

        minimum_tokens = context_length + 1
        if token_ids.numel() < minimum_tokens:
            raise ValueError(
                "token_ids must contain at least context_length + 1 tokens"
            )

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.example_count = 1 + (token_ids.numel() - minimum_tokens) // stride

    def __len__(self) -> int:
        """Return how many complete training windows are available."""
        return self.example_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one input/target pair for next-token prediction."""
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} is outside this dataset")

        start = index * self.stride
        stop = start + self.context_length + 1
        window = self.token_ids[start:stop]

        return window[:-1], window[1:]
