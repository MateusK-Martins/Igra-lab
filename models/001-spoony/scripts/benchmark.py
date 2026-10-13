"""Measure the configured real-data training path without saving trained weights."""

import json
import time
from dataclasses import asdict
from itertools import cycle

import torch

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.benchmark import MEASURED_UPDATES, WARMUP_UPDATES
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config, training_config
from src.data.dataloader import build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler
from src.training.step import train_step
from src.training.validation import validate


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    data = preparation_config.output_directory
    tokenizer = ByteLevelBPETokenizer.load(data / "tokenizer.json")
    model, _ = build_model(tokenizer)
    model.to(device)
    loader = build_split_dataloader(
        data / "train", chunking_config, train_loader_config
    )
    validation = build_split_dataloader(
        data / "validation", chunking_config, validation_loader_config
    )
    context, scaler = build_precision(device, precision_config)
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)
    start = time.perf_counter()
    initial_loss, _ = validate(model, validation, context, device)
    validation_seconds = time.perf_counter() - start
    iterator = cycle(loader)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(WARMUP_UPDATES + MEASURED_UPDATES):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, updated = train_step(
            model,
            [next(iterator)],
            optimizer,
            scheduler,
            context,
            scaler,
            device,
            training_config.max_grad_norm,
        )
        torch.cuda.synchronize()
        if not updated:
            raise RuntimeError("Benchmark update skipped")
        if index >= WARMUP_UPDATES:
            times.append(time.perf_counter() - start)
    report = {
        "scheduler": asdict(scheduler_config),
        "optimizer": asdict(optimizer_config),
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size,
        "initial_validation_loss": initial_loss,
        "validation_seconds": validation_seconds,
        "measured_updates": len(times),
        "mean_update_seconds": sum(times) / len(times),
        "p95_update_seconds": sorted(times)[int(len(times) * 0.95) - 1],
        "tokens_per_update": train_loader_config.batch_size
        * chunking_config.context_length,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "benchmark_final_loss": loss,
        "torch_cpu_threads": torch.get_num_threads(),
    }
    path = logging_config.directory.parent / "benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
