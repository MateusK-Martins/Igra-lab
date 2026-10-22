"""Small JSONL instruction datasets with response-only supervision."""

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.tokenizer import Tokenizer
from src.sft.config import SFTDataConfig
from src.training.loss import IGNORE_INDEX


class SFTDataset(Dataset[tuple[Tensor, Tensor]]):
    """One instruction/response per window; reject oversized examples.

    Encode prompt and response separately to make their token boundary explicit.
    Right padding cannot affect earlier real tokens in causal attention. Padding
    and prompt labels are ignored; the response and its end token are supervised.
    Examples are held in RAM; no packing or silent truncation is performed.
    """

    def __init__(self, path: Path, tokenizer: Tokenizer, config: SFTDataConfig) -> None:
        end_id = tokenizer.token_to_id(config.end_token)
        self.examples: list[tuple[Tensor, Tensor]] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or any(
                    not isinstance(record.get(key), str) or not record[key].strip()
                    for key in ("instruction", "response")
                ):
                    raise ValueError(
                        f"{path}:{line_number}: instruction/response must be non-empty strings"
                    )
                prompt = tokenizer.encode(
                    config.instruction_prefix
                    + record["instruction"]
                    + config.response_prefix
                )
                response = tokenizer.encode(record["response"]) + [end_id]
                if not prompt:
                    raise ValueError("SFT prompt must contain tokens")
                sequence = prompt + response
                if len(sequence) - 1 > config.context_length:
                    raise ValueError(
                        f"{path}:{line_number}: example exceeds context_length"
                    )
                inputs = torch.full((config.context_length,), end_id, dtype=torch.long)
                targets = torch.full_like(inputs, IGNORE_INDEX)
                inputs[: len(sequence) - 1] = torch.tensor(sequence[:-1])
                targets[len(prompt) - 1 : len(sequence) - 1] = torch.tensor(response)
                self.examples.append((inputs, targets))
        if not self.examples:
            raise ValueError("SFT dataset must contain examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        inputs, targets = self.examples[index]
        return inputs.clone(), targets.clone()


def collate_instructions(examples):
    inputs, targets = zip(*examples, strict=True)
    targets = torch.stack(targets)
    last = (targets != IGNORE_INDEX).nonzero()[:, 1].max().item() + 1
    return torch.stack(inputs)[:, :last], targets[:, :last]
