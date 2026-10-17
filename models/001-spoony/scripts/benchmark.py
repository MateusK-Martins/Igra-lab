"""Benchmark real-data optimizer updates; discard all benchmark weights."""

import json
import math
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.benchmark import (
    MEASURED_UPDATES,
    MICRO_BATCH_SIZE,
    WARMUP_UPDATES,
)
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.scheduler import scheduler_config
from configs.training.training import training_config
from src.config.paths import results_path
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
    benchmark_loader_config = train_loader_config
    benchmark_training_config = training_config
    if MICRO_BATCH_SIZE is not None:
        effective_batch = (
            train_loader_config.batch_size * training_config.accumulation_steps
        )
        if effective_batch % MICRO_BATCH_SIZE:
            raise ValueError("Benchmark microbatch must divide effective batch")
        benchmark_loader_config = replace(
            train_loader_config, batch_size=MICRO_BATCH_SIZE
        )
        benchmark_training_config = replace(
            training_config, accumulation_steps=effective_batch // MICRO_BATCH_SIZE
        )
    loader = build_split_dataloader(
        data / "train", chunking_config, benchmark_loader_config
    )
    validation = build_split_dataloader(
        data / "validation",
        chunking_config,
        replace(
            validation_loader_config, batch_size=benchmark_loader_config.batch_size
        ),
    )
    context, scaler = build_precision(device, precision_config)
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)
    iterator = iter(loader)
    times, token_counts = [], []
    torch.cuda.reset_peak_memory_stats()
    for index in range(WARMUP_UPDATES + MEASURED_UPDATES):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batches = [
            next(iterator) for _ in range(benchmark_training_config.accumulation_steps)
        ]
        loss, updated = train_step(
            model,
            batches,
            optimizer,
            scheduler,
            context,
            scaler,
            device,
            training_config.max_grad_norm,
        )
        torch.cuda.synchronize()
        if not updated or not math.isfinite(loss):
            raise RuntimeError("Benchmark update skipped or loss is nonfinite")
        if index >= WARMUP_UPDATES:
            times.append(time.perf_counter() - start)
            token_counts.append(sum(target.numel() for _, target in batches))
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()
    start = time.perf_counter()
    validation_loss, _ = validate(model, validation, context, device)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    # Verify that these full-size weights survive serialization and reassembly.
    sample = next(iter(validation))[0][:1, :16].to(device)
    model.eval()
    with torch.no_grad(), context:
        expected = model(sample).float().cpu()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        torch.save(model.state_dict(), path)
        restored, _ = build_model(tokenizer)
        restored.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        restored.to(device).eval()
        with torch.no_grad(), context:
            actual = restored(sample).float().cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rate = sum(token_counts) / sum(times)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": chunking_config.context_length,
        "loader": asdict(benchmark_loader_config),
        "training": asdict(benchmark_training_config),
        "precision": asdict(precision_config),
        "scheduler": asdict(scheduler_config),
        "measured_updates": len(times),
        "tokens_per_update": token_counts[0],
        "mean_update_seconds": sum(times) / len(times),
        "tokens_per_second": rate,
        "peak_allocated_vram_bytes": peak,
        "final_benchmark_loss": loss,
        "validation_loss": validation_loss,
        "full_validation_seconds": validation_seconds,
        "training_only_seconds_for_600m_tokens": 600_000_000 / rate,
        "weights_save_load_logits_exact": True,
        "benchmark_weights_discarded": True,
    }
    path = results_path("data/wikipedia-en-600m/benchmark-2048.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
