import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts import train
from src.config.errors import ConfigError
from src.config.types import (
    CheckpointConfig,
    ChunkingConfig,
    DataLoaderConfig,
    LoggingConfig,
    PrecisionConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.data.serialization import TokenStore
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.logging import build_writer


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(8, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, inputs):
        return self.head(self.dropout(self.embedding(inputs)))


@pytest.fixture
def configured_script(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "qa_config", None)
    data = tmp_path / "data"
    tokenizer = ByteLevelBPETokenizer.train(
        ["A fox found a bird."],
        target_vocab_size=270,
        min_frequency=1,
        special_tokens=("<|unk|>", "<|eot|>"),
    )
    tokenizer.save(data / "tokenizer.json")
    from src.training.manifest import file_sha256

    for split in ("train", "validation"):
        TokenStore(data / split).write_documents(
            [("doc", list(range(1, 26)))],
            tokenizer_sha256=file_sha256(data / "tokenizer.json"),
        )
    monkeypatch.setattr(
        train,
        "preparation_config",
        replace(train.preparation_config, output_directory=data),
    )
    monkeypatch.setattr(train, "run_config", RunConfig("cpu"))
    monkeypatch.setattr(train, "precision_config", PrecisionConfig("fp32"))
    monkeypatch.setattr(train, "chunking_config", ChunkingConfig(4, 4))
    for name, shuffle in (
        ("train_loader_config", True),
        ("validation_loader_config", False),
    ):
        monkeypatch.setattr(
            train, name, DataLoaderConfig(2, shuffle, 0, False, False, 42)
        )
    monkeypatch.setattr(train, "training_config", TrainingConfig(1, 1.0, 2))
    monkeypatch.setattr(train, "scheduler_config", SchedulerConfig(0, 3, 0.1))
    monkeypatch.setattr(
        train, "logging_config", LoggingConfig(tmp_path / "run/tensorboard", 1)
    )
    monkeypatch.setattr(
        train, "checkpoint_config", CheckpointConfig(tmp_path / "run/checkpoints", 1)
    )
    models, writers, resume_steps = [], [], []

    def model_factory(tokenizer):
        model = TinyModel(tokenizer.vocab_size)
        models.append(model)
        return model, []

    def writer_factory(config, *, resume_step=None):
        writer = build_writer(config, resume_step=resume_step)
        writers.append(writer)
        resume_steps.append(resume_step)
        return writer

    monkeypatch.setattr(train, "build_model", model_factory)
    monkeypatch.setattr(train, "build_writer", writer_factory)
    return tmp_path, models, writers, resume_steps


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def test_fresh_script_runs_real_data_training_logging_and_checkpoint(configured_script):
    root, models, writers, resume_steps = configured_script
    train.main()
    manifest = json.loads((root / "run/manifest.json").read_text())
    checkpoint = torch.load(
        root / "run/checkpoints/step-00000003.pt", weights_only=True
    )
    assert checkpoint["settings"] == manifest
    assert "resume_from" not in manifest["settings"]
    assert checkpoint["training_state"]["updates"] == 3
    assert checkpoint["training_state"]["completed_epochs"] == 1
    assert checkpoint["training_state"]["tokens"] == 24
    assert models[0].embedding.weight is models[0].head.weight
    assert writers[0].file_writer is None
    assert resume_steps == [None]
    events = EventAccumulator(str(root / "run/tensorboard")).Reload()
    assert [e.step for e in events.Scalars("train/loss")] == [1, 2, 3]
    assert [e.step for e in events.Scalars("validation/loss")] == [2, 3]


def test_script_resume_matches_uninterrupted_training(configured_script, monkeypatch):
    from src.training import loop

    root, _, writers, resume_steps = configured_script
    torch.manual_seed(123)
    train.main()
    expected = torch.load(root / "run/checkpoints/step-00000003.pt", weights_only=True)
    monkeypatch.setattr(
        train,
        "logging_config",
        replace(train.logging_config, directory=root / "resume/tensorboard"),
    )
    monkeypatch.setattr(
        train,
        "checkpoint_config",
        replace(train.checkpoint_config, directory=root / "resume/checkpoints"),
    )
    save = loop.save_checkpoint

    def interrupt(path, **kwargs):
        save(path, **kwargs)
        raise RuntimeError("interrupted after checkpoint")

    torch.manual_seed(123)
    with monkeypatch.context() as patch:
        patch.setattr(loop, "save_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            train.main()
    assert writers[-1].file_writer is None
    monkeypatch.setattr(
        train,
        "run_config",
        RunConfig("cpu", root / "resume/checkpoints/step-00000001.pt"),
    )
    torch.manual_seed(999)
    train.main()
    actual = torch.load(root / "resume/checkpoints/step-00000003.pt", weights_only=True)
    assert_tree_equal(actual, expected)
    assert resume_steps == [None, None, 1]
    assert all(writer.file_writer is None for writer in writers)


def test_writer_closed_when_training_fails(configured_script, monkeypatch):
    _, _, writers, _ = configured_script
    monkeypatch.setattr(
        train, "train_loop", MagicMock(side_effect=RuntimeError("training failed"))
    )
    with pytest.raises(RuntimeError, match="training failed"):
        train.main()
    assert len(writers) == 1
    assert writers[0].file_writer is None


def test_bad_resume_creates_no_manifest_or_writer(configured_script, monkeypatch):
    root, _, writers, _ = configured_script
    monkeypatch.setattr(train, "run_config", RunConfig("cpu", root / "missing.pt"))
    with pytest.raises(FileNotFoundError):
        train.main()
    assert not (root / "run").exists()
    assert not writers


@pytest.mark.parametrize("device", ["mps", "cuda:0", None])
def test_run_config_rejects_unsupported_device(device):
    with pytest.raises(ConfigError, match="device"):
        RunConfig(device)


@pytest.mark.parametrize("path", ["checkpoint.pt", 1, False])
def test_run_config_rejects_nonpath_checkpoint(path):
    with pytest.raises(ConfigError, match="resume_from"):
        RunConfig("cpu", path)


def test_train_script_combines_existing_story_and_qa_datasets(
    configured_script, monkeypatch
):
    from torch.utils.data import ConcatDataset

    from src.sft.config import SFTDataConfig

    root, _, _, _ = configured_script
    qa_path = root / "qa.jsonl"
    qa_path.write_text(json.dumps({"instruction": "a", "response": "b"}) + "\n")
    monkeypatch.setattr(
        train,
        "qa_config",
        SFTDataConfig(
            train_path=qa_path,
            validation_path=qa_path,
            tokenizer_path=root / "data/tokenizer.json",
            context_length=4,
            instruction_prefix="a",
            response_prefix="b",
        ),
    )
    datasets = []
    original = train.build_dataloader

    def capture(dataset, config):
        datasets.append(dataset)
        return original(dataset, config)

    monkeypatch.setattr(train, "build_dataloader", capture)
    train.main()
    assert isinstance(datasets[0], ConcatDataset)
    assert [len(dataset) for dataset in datasets[0].datasets] == [6, 1]
    manifest = json.loads((root / "run/manifest.json").read_text())
    assert (
        manifest["artifacts"]["qa-train.jsonl"]
        == manifest["artifacts"]["qa-validation.jsonl"]
    )
