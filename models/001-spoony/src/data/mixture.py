"""Deterministic whole-batch mixing with separate supervision densities."""

import math

import torch
from torch.utils.data import Dataset


class BatchMixtureDataset(Dataset):
    def __init__(
        self, stories, qa, *, batch_size: int, story_batches_per_qa: int, seed: int
    ):
        if not len(stories) or not len(qa):
            raise ValueError("Both datasets must be nonempty")
        if batch_size <= 0 or story_batches_per_qa <= 0:
            raise ValueError("Batch size and mixture ratio must be positive")
        self.stories, self.qa = stories, qa
        self.batch_size, self.ratio = batch_size, story_batches_per_qa
        generator = torch.Generator().manual_seed(seed)
        self.story_order = torch.randperm(len(stories), generator=generator)
        self.qa_order = torch.randperm(len(qa), generator=generator)
        self.cycles = max(
            math.ceil(len(stories) / (batch_size * self.ratio)),
            math.ceil(len(qa) / batch_size),
        )

    def __len__(self):
        return self.cycles * (self.ratio + 1) * self.batch_size

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        batch, slot = divmod(index, self.batch_size)
        cycle, kind = divmod(batch, self.ratio + 1)
        if kind == self.ratio:
            return self.qa[
                int(self.qa_order[(cycle * self.batch_size + slot) % len(self.qa)])
            ]
        offset = cycle * self.ratio * self.batch_size + kind * self.batch_size + slot
        return self.stories[int(self.story_order[offset % len(self.stories)])]
