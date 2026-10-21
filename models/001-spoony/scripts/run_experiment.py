"""Train the configured recipe and persist a compact final experiment report."""

import json
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config
from scripts.train import main as train_main
from src.data.tokenizer import ByteLevelBPETokenizer


def main():
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_main()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    root = logging_config.directory.parent
    path = checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = ByteLevelBPETokenizer.load(
        preparation_config.output_directory / "tokenizer.json"
    )
    model, _ = build_model(tokenizer)
    model.load_state_dict(saved["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Final checkpoint contains nonfinite weights")
    events = EventAccumulator(str(logging_config.directory)).Reload()
    curve = [
        {"update": e.step, "loss": e.value} for e in events.Scalars("validation/loss")
    ]
    saved_steps = {
        int(p.stem.split("-")[1]) for p in checkpoint_config.directory.glob("step-*.pt")
    }
    best = min(
        (row for row in curve if row["update"] in saved_steps),
        key=lambda row: row["loss"],
    )
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds_including_loading_validation_saving_and_export": elapsed,
        "peak_allocated_vram_bytes": peak,
        "gpu": torch.cuda.get_device_name(),
        "seed": 42,
        "training_state": {
            k: v
            for k, v in saved["training_state"].items()
            if not isinstance(v, torch.Tensor)
        },
        "validation_curve": curve,
        "best_saved_validation": best,
        "best_checkpoint": f"checkpoints/step-{best['update']:08d}.pt",
        "final_checkpoint": str(path.relative_to(root)),
        "benchmark": json.loads((root / "benchmark.json").read_text()),
        "comparison": "Same pinned Wikipedia artifacts, context, microbatch, accumulation, effective batch, optimizer, scheduler, seed and full one-pass training exposure as the completed Lantern 70M baseline.",
    }
    (root / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
