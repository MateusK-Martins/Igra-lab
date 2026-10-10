import copy
import json
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from test_train_script import TinyModel, assert_tree_equal
from torch.nn import functional as F

from scripts import sft
from src.config.errors import ConfigError
from src.config.types import (
    CheckpointConfig,
    DataLoaderConfig,
    LoggingConfig,
    PrecisionConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.data.tokenizer import ByteLevelBPETokenizer
from src.sft.config import SFTDataConfig, SFTInitializationConfig
from src.sft.data import SFTDataset
from src.training.checkpoint import load_model_weights
from src.training.manifest import file_sha256
from src.training.scheduler import build_scheduler
from src.training.step import train_step
from src.training.validation import validate


@pytest.fixture
def sft_data(tmp_path):
    tokenizer = ByteLevelBPETokenizer.train(
        ["User: Who?\nAssistant: A fox. A bird."],
        target_vocab_size=280,
        min_frequency=1,
        special_tokens=("<|unk|>", "<|eot|>"),
    )
    tokenizer.save(tmp_path / "tokenizer.json")
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(
                json.dumps({"instruction": "Who?", "response": text})
                for text in ("A fox.", "A bird.", "A fox. A bird.")
            )
            + "\n"
        )
    config = SFTDataConfig(
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "tokenizer.json",
        64,
    )
    return tokenizer, config


def test_dataset_shift_mask_end_token_and_padding(sft_data):
    tokenizer, config = sft_data
    dataset = SFTDataset(config.train_path, tokenizer, config)
    inputs, targets = dataset[0]
    prompt = tokenizer.encode("User: Who?\nAssistant: ")
    response = tokenizer.encode("A fox.") + [tokenizer.token_to_id("<|eot|>")]
    assert inputs.shape == targets.shape == (64,)
    assert inputs[: len(prompt)].tolist() == prompt
    assert (
        inputs[len(prompt) : len(prompt) + len(response) - 1].tolist() == response[:-1]
    )
    assert (targets[: len(prompt) - 1] == -100).all()
    assert (
        targets[len(prompt) - 1 : len(prompt) + len(response) - 1].tolist() == response
    )
    assert (targets[len(prompt) + len(response) - 1 :] == -100).all()
    inputs.zero_()
    assert not torch.equal(inputs, dataset[0][0])


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"instruction": "", "response": "a"},
        {"instruction": "a", "response": None},
        [],
    ],
)
def test_bad_records_rejected(sft_data, record):
    tokenizer, config = sft_data
    config.train_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="instruction/response"):
        SFTDataset(config.train_path, tokenizer, config)


def test_empty_and_oversized_datasets_rejected(sft_data):
    tokenizer, config = sft_data
    with pytest.raises(ValueError, match="context_length"):
        SFTDataset(config.train_path, tokenizer, replace(config, context_length=1))
    config.train_path.write_text("\n")
    with pytest.raises(ValueError, match="contain examples"):
        SFTDataset(config.train_path, tokenizer, config)


@pytest.mark.parametrize(
    "change",
    [
        {"context_length": 0},
        {"context_length": True},
        {"train_path": "bad"},
        {"response_prefix": ""},
    ],
)
def test_data_config_validation(sft_data, change):
    _, config = sft_data
    with pytest.raises(ConfigError):
        replace(config, **change)


def test_initialization_config_requires_path():
    with pytest.raises(ConfigError):
        SFTInitializationConfig("bad")


def test_masked_accumulation_and_validation_match_direct_supervised_loss():
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 5).double()
    reference = copy.deepcopy(model)
    batches = [
        (torch.randn(1, 4, 3, dtype=torch.float64), torch.tensor([[-100, 1, 2, 3]])),
        (
            torch.randn(1, 4, 3, dtype=torch.float64),
            torch.tensor([[-100, -100, -100, 4]]),
        ),
        (torch.randn(1, 4, 3, dtype=torch.float64), torch.full((1, 4), -100)),
    ]
    inputs, targets = (torch.cat([batch[i] for batch in batches]) for i in (0, 1))
    logits = reference(inputs)
    logits.retain_grad()
    expected = F.cross_entropy(logits.reshape(-1, 5), targets.flatten())
    expected.backward()
    assert torch.count_nonzero(logits.grad[targets == -100]) == 0
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 100)
    reference_opt.step()
    scheduler = build_scheduler(opt, SchedulerConfig(0, 3, 0.1))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss, updated = train_step(
        model, batches, opt, scheduler, nullcontext(), scaler, torch.device("cpu"), 100
    )
    assert updated and loss == pytest.approx(expected.item())
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted)
    val_expected = F.cross_entropy(model(inputs).reshape(-1, 5), targets.flatten())
    val, _ = validate(model, batches, nullcontext(), torch.device("cpu"))
    assert val == pytest.approx(val_expected.item())
    assert model.training
    empty = [batches[-1]]
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="target tokens"):
        train_step(
            model,
            empty,
            opt,
            scheduler,
            nullcontext(),
            scaler,
            torch.device("cpu"),
            100,
        )
    assert_tree_equal(model.state_dict(), state)
    with pytest.raises(ValueError, match="target tokens"):
        validate(model, empty, nullcontext(), torch.device("cpu"))
    assert model.training


@pytest.fixture
def configured_sft(sft_data, tmp_path, monkeypatch):
    tokenizer, config = sft_data
    base = TinyModel(tokenizer.vocab_size)
    checkpoint = tmp_path / "base.pt"
    torch.save(
        {
            "format_version": 1,
            "settings": {
                "artifacts": {"tokenizer.json": file_sha256(config.tokenizer_path)}
            },
            "model": base.state_dict(),
            "optimizer": {"old": True},
            "training_state": {"updates": 2400},
        },
        checkpoint,
    )
    monkeypatch.setattr(sft, "data_config", config)
    monkeypatch.setattr(
        sft, "initialization_config", SFTInitializationConfig(checkpoint)
    )
    monkeypatch.setattr(sft, "run_config", RunConfig("cpu"))
    monkeypatch.setattr(sft, "precision_config", PrecisionConfig("fp32"))
    monkeypatch.setattr(sft, "training_config", TrainingConfig(1, 1.0, 2))
    monkeypatch.setattr(sft, "scheduler_config", SchedulerConfig(0, 3, 0.1))
    for name, shuffle in (
        ("train_loader_config", True),
        ("validation_loader_config", False),
    ):
        monkeypatch.setattr(
            sft, name, DataLoaderConfig(2, shuffle, 0, False, False, 42)
        )
    monkeypatch.setattr(
        sft, "logging_config", LoggingConfig(tmp_path / "run/tensorboard", 1)
    )
    monkeypatch.setattr(
        sft, "checkpoint_config", CheckpointConfig(tmp_path / "run/checkpoints", 1)
    )
    monkeypatch.setattr(
        sft, "build_model", lambda tokenizer: (TinyModel(tokenizer.vocab_size), [])
    )
    return tmp_path, base


def test_weights_only_initialization_and_tokenizer_guard(configured_sft):
    root, base = configured_sft
    model = TinyModel(base.embedding.num_embeddings)
    rng = torch.get_rng_state().clone()
    load_model_weights(
        root / "base.pt", model=model, tokenizer_path=sft.data_config.tokenizer_path
    )
    assert_tree_equal(model.state_dict(), base.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    sft.data_config.tokenizer_path.write_text("different")
    with pytest.raises(ValueError, match="Tokenizer"):
        load_model_weights(
            root / "base.pt", model=model, tokenizer_path=sft.data_config.tokenizer_path
        )
    assert_tree_equal(model.state_dict(), base.state_dict())


def test_sft_fresh_and_exact_resume_without_base_checkpoint(
    configured_sft, monkeypatch
):
    from src.training import loop

    root, _ = configured_sft
    torch.manual_seed(123)
    sft.main()
    expected = torch.load(root / "run/checkpoints/step-00000003.pt", weights_only=True)
    assert expected["training_state"]["updates"] == 3
    assert expected["settings"]["settings"]["stage"] == "sft"
    assert expected["settings"]["artifacts"]["initial_checkpoint"] == file_sha256(
        root / "base.pt"
    )
    assert "old" not in expected["optimizer"]
    monkeypatch.setattr(
        sft, "logging_config", LoggingConfig(root / "resume/tensorboard", 1)
    )
    monkeypatch.setattr(
        sft, "checkpoint_config", CheckpointConfig(root / "resume/checkpoints", 1)
    )
    save = loop.save_checkpoint

    def interrupt(path, **kwargs):
        save(path, **kwargs)
        raise RuntimeError("interrupted")

    torch.manual_seed(123)
    with monkeypatch.context() as patch:
        patch.setattr(loop, "save_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            sft.main()
    (root / "base.pt").unlink()
    monkeypatch.setattr(
        sft,
        "run_config",
        RunConfig("cpu", root / "resume/checkpoints/step-00000001.pt"),
    )
    torch.manual_seed(999)
    sft.main()
    actual = torch.load(root / "resume/checkpoints/step-00000003.pt", weights_only=True)
    assert_tree_equal(actual, expected)


def test_sft_rejects_changed_data_on_resume(configured_sft, monkeypatch):
    root, _ = configured_sft
    sft.main()
    sft.data_config.train_path.write_text(
        json.dumps({"instruction": "Other?", "response": "A fox."})
    )
    monkeypatch.setattr(
        sft, "run_config", RunConfig("cpu", root / "run/checkpoints/step-00000001.pt")
    )
    with pytest.raises(ValueError, match="settings"):
        sft.main()


def test_configured_model_accepts_sft_and_right_padding_is_causally_isolated(sft_data):
    from configs.model.model import build_model
    from src.training.loss import next_token_loss

    tokenizer, config = sft_data
    inputs, targets = SFTDataset(config.train_path, tokenizer, config)[0]
    model, _ = build_model(tokenizer)
    model.eval()
    length = int((targets != -100).nonzero()[-1]) + 1
    logits = model(inputs.unsqueeze(0))
    with torch.no_grad():
        shorter_logits = model(inputs[:length].unsqueeze(0))
    torch.testing.assert_close(logits[:, :length], shorter_logits, rtol=1e-4, atol=1e-5)
    loss = next_token_loss(logits, targets.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embedding.weight.grad is not None
