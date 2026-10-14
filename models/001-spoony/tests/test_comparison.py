from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.comparison import ComparisonConfig
from src.config.types import LoggingConfig, TrainingConfig
from src.data.mixture import BatchMixtureDataset
from src.evaluation.qa import answer_scores, normalize_answer
from src.training.loop import train_loop


def test_batch_mixture_ratio_determinism_and_whole_batches():
    stories = TensorDataset(torch.arange(13) + 100)
    qa = TensorDataset(torch.arange(7))
    first = BatchMixtureDataset(
        stories, qa, batch_size=2, story_batches_per_qa=3, seed=42
    )
    second = BatchMixtureDataset(
        stories, qa, batch_size=2, story_batches_per_qa=3, seed=42
    )
    assert [first[i][0].item() for i in range(len(first))] == [
        second[i][0].item() for i in range(len(second))
    ]
    for batch in range(len(first) // 2):
        values = [first[batch * 2 + offset][0].item() for offset in range(2)]
        assert (
            all(value < 100 for value in values)
            if batch % 4 == 3
            else all(value >= 100 for value in values)
        )
    with pytest.raises(IndexError):
        first[len(first)]


def test_qa_official_style_normalization_multiple_aliases_and_partial_f1():
    assert normalize_answer(" The, U.S.  President! ") == "us president"
    assert answer_scores("The United States", ["USA", "United States"]) == (1, 1)
    exact, f1 = answer_scores("red blue", ["red green"])
    assert exact == 0 and f1 == pytest.approx(0.5)
    assert answer_scores("unrelated", ["answer"]) == (0, 0)


def test_training_counts_processed_and_supervised_tokens_separately(tmp_path):
    inputs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 0], [-100, -100, 3, 0]])
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=1,
        generator=torch.Generator().manual_seed(42),
    )
    model = nn.Sequential(nn.Embedding(4, 3), nn.Linear(3, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = train_loop(
        model,
        loader,
        loader,
        optimizer,
        scheduler,
        nullcontext(),
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        TrainingConfig(1, 1.0, 2),
        2,
        SimpleNamespace(add_scalar=lambda *args: None),
        LoggingConfig(Path(tmp_path), 1),
    )
    assert state.tokens == 8
    assert state.supervised_tokens == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_seconds": 0},
        {"story_batches_per_qa": True},
        {"budget_seconds": 100, "reserve_seconds": 100},
    ],
)
def test_invalid_comparison_config(kwargs):
    with pytest.raises(ValueError):
        ComparisonConfig(**kwargs)


def test_qa_generation_uses_prompt_only_and_stops_at_end_token(tmp_path):
    import json

    from src.evaluation.qa import evaluate_answers

    path = tmp_path / "qa.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q",
                "instruction": "question",
                "question": "question",
                "answers": ["blue"],
            }
        )
        + "\n"
    )

    class Tokenizer:
        def encode(self, text):
            return [2]

        def token_to_id(self, text):
            return 0

        def decode(self, ids):
            return "blue" if ids == [1] else "wrong"

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, *, cache=None):
            assert not torch.is_grad_enabled()
            self.seen.append(x.tolist())
            logits = torch.zeros(1, x.shape[1], 3)
            logits[0, -1, 1 if len(self.seen) == 1 else 0] = 1
            return logits

    model = Model()
    config = SimpleNamespace(
        validation_path=path,
        instruction_prefix="Context: ",
        response_prefix="Answer: ",
        context_length=8,
        end_token="<|eot|>",
    )
    result = evaluate_answers(
        model,
        [],
        Tokenizer(),
        config,
        ComparisonConfig(qa_generation_examples=1),
        nullcontext(),
        torch.device("cpu"),
    )
    assert result["exact_match"] == 100 and result["f1"] == 100
    assert model.seen == [[[2]], [[1]]]
    assert model.training
