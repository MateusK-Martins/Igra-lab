import json
import time

import torch

from configs.data.dataloader import chunking_config
from configs.model.model import build_model
from configs.training.comparison import comparison_config
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config, training_config
from scripts.comparison_common import loaders
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler
from src.training.step import train_step
from src.training.validation import validate


def main():
    torch.set_num_threads(comparison_config.cpu_threads)
    tokenizer, train, stories, qa = loaders()
    device = torch.device("cuda")
    model, _ = build_model(tokenizer)
    model.to(device)
    context, scaler = build_precision(device, precision_config)
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)
    start = time.perf_counter()
    story_loss, _ = validate(model, stories, context, device)
    qa_loss, _ = validate(model, qa, context, device)
    validation_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(train)
    times = []
    for index in range(
        comparison_config.benchmark_warmup + comparison_config.benchmark_steps
    ):
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
            raise RuntimeError("Benchmark skipped update")
        if index >= comparison_config.benchmark_warmup:
            times.append(time.perf_counter() - start)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": tokenizer.vocab_size,
        "gpu": torch.cuda.get_device_name(),
        "mean_update_seconds": sum(times) / len(times),
        "p95_update_seconds": sorted(times)[int(len(times) * 0.95) - 1],
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "validation_seconds_both_sets": validation_seconds,
        "initial_story_loss": story_loss,
        "initial_qa_answer_loss": qa_loss,
        "last_benchmark_loss": loss,
        "token_positions_per_update": train.batch_size * chunking_config.context_length,
        "benchmark_discards_weights": True,
    }
    root = logging_config.directory.parent
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
