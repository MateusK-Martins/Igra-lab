"""Checkpoints between optimizer updates, for deterministic map-style data."""

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from src.config.snapshot import normalize_settings
from src.training.state import TrainingState


def loader_signature(loader: DataLoader) -> dict:
    """Reject loaders for which replay cannot reproduce this project's batches."""
    if loader.num_workers != 0:
        raise ValueError("Checkpoint resume currently requires num_workers=0")
    if loader.generator is None or loader.generator.device.type != "cpu":
        raise ValueError(
            "Checkpoint resume requires an explicit CPU DataLoader generator"
        )
    if type(loader.batch_sampler) is not BatchSampler or type(loader.sampler) not in (
        RandomSampler,
        SequentialSampler,
    ):
        raise ValueError("Checkpoint resume requires standard batching and sampling")
    if isinstance(loader.sampler, RandomSampler) and (
        loader.sampler.replacement or loader.sampler.generator is not loader.generator
    ):
        raise ValueError(
            "Checkpoint resume requires shuffle without replacement using the loader generator"
        )
    return {
        "dataset_length": len(loader.dataset),
        "batches": len(loader),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "shuffle": isinstance(loader.sampler, RandomSampler),
    }


def _validate_state(state: TrainingState, loader: DataLoader) -> None:
    for name in (
        "updates",
        "batches",
        "tokens",
        "supervised_tokens",
        "completed_epochs",
        "batches_in_epoch",
    ):
        value = getattr(state, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid training state: {name}")
    if state.supervised_tokens > state.tokens:
        raise ValueError("Supervised tokens exceed processed token positions")
    if len(loader) == 0 or state.batches_in_epoch >= len(loader):
        raise ValueError("Invalid batches_in_epoch")
    if state.batches != state.completed_epochs * len(loader) + state.batches_in_epoch:
        raise ValueError("Training batch/epoch counters do not match the DataLoader")
    if state.updates > state.batches:
        raise ValueError("Training updates exceed consumed batches")
    if state.batches_in_epoch:
        _validate_rng(state.epoch_generator_state)
    elif state.epoch_generator_state is not None:
        raise ValueError("An epoch boundary must not retain epoch_generator_state")


def _validate_rng(value) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise ValueError("Invalid random generator state")
    torch.Generator().set_state(value)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: dict[str, object],
) -> None:
    """Save atomically after an update's validation/logging, with no pending grads.

    settings must include the scheduler recipe and all choices affecting the
    model, data and updates; include tokenizer/dataset hashes as run metadata.
    """
    train_signature = loader_signature(train_loader)
    validation_signature = loader_signature(validation_loader)
    _validate_state(state, train_loader)
    payload = {
        "format_version": 1,
        "settings": normalize_settings(settings),
        "model": model.state_dict(),
        "model_training": model.training,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "training_state": asdict(state),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_loader.generator.get_state(),
        "validation_generator": validation_loader.generator.get_state(),
        "train_loader": train_signature,
        "validation_loader": validation_signature,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary_path = Path(file.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: dict[str, object],
) -> TrainingState:
    """Build components first, then load; pass returned state to train_loop.

    Requires the same settings, deterministic dataset/collation and CUDA device
    count. Exact replay currently supports only num_workers=0. Python/NumPy RNG
    and stochastic data transforms are outside this contract.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
    ):
        raise ValueError("Unsupported checkpoint format_version")
    if payload["settings"] != normalize_settings(settings):
        raise ValueError("Checkpoint settings do not match this run")
    for name, loader in (
        ("train_loader", train_loader),
        ("validation_loader", validation_loader),
    ):
        if payload[name] != loader_signature(loader):
            raise ValueError(f"Checkpoint {name} does not match this run")
    state = TrainingState(**payload["training_state"])
    _validate_state(state, train_loader)
    for name in ("torch_rng", "train_generator", "validation_generator"):
        _validate_rng(payload[name])
    cuda_states = payload["cuda_rng"]
    if (
        not isinstance(cuda_states, list)
        or len(cuda_states) != torch.cuda.device_count()
    ):
        raise ValueError("Checkpoint CUDA device count does not match this run")
    model.load_state_dict(payload["model"])
    model.train(payload["model_training"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    train_loader.generator.set_state(payload["train_generator"])
    validation_loader.generator.set_state(payload["validation_generator"])
    torch.set_rng_state(payload["torch_rng"])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return state


def training_iterator(loader: DataLoader, state: TrainingState):
    """Recreate a partial epoch without advancing training RNG or loader RNG."""
    if state.batches_in_epoch == 0:
        state.epoch_generator_state = (
            loader.generator.get_state() if loader.generator is not None else None
        )
        return iter(loader)
    loader_signature(loader)
    _validate_state(state, loader)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    generator_state = loader.generator.get_state()
    try:
        loader.generator.set_state(state.epoch_generator_state)
        iterator = iter(loader)
        for _ in range(state.batches_in_epoch):
            next(iterator)
    finally:
        loader.generator.set_state(generator_state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    return iterator


def load_model_weights(path: Path, *, model: nn.Module, tokenizer_path: Path) -> None:
    """Start a new stage from weights, without restoring optimizer/RNG/counters."""
    from src.training.manifest import file_sha256

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    expected = payload.get("settings", {}).get("artifacts", {}).get("tokenizer.json")
    if expected != file_sha256(tokenizer_path):
        raise ValueError("Tokenizer does not match the initialization checkpoint")
    model.load_state_dict(payload["model"])
