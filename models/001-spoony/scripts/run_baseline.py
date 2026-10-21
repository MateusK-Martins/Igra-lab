"""Run the frozen recipe, then save timings, metrics and fixed greedy samples."""

import json
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configs.data.corpus import RAW_DIRECTORY
from configs.data.dataloader import chunking_config
from configs.inference_baseline import MAX_NEW_TOKENS, PROMPTS
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
    checkpoint_path = (
        checkpoint_config.directory / f"step-{scheduler_config.total_steps:08d}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    events = EventAccumulator(str(logging_config.directory)).Reload()
    curve = [
        {"update": event.step, "loss": event.value}
        for event in events.Scalars("validation/loss")
    ]
    saved_steps = {
        int(path.stem.split("-")[1])
        for path in checkpoint_config.directory.glob("step-*.pt")
    }
    best = min(
        (row for row in curve if row["update"] in saved_steps),
        key=lambda row: row["loss"],
    )
    report = {
        "branch": "codex/lantern-30m-tokenmlp-projection-only",
        "training_seconds_including_validation_and_saving": elapsed,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "training_state": {
            key: value
            for key, value in checkpoint["training_state"].items()
            if not isinstance(value, torch.Tensor)
        },
        "validation_curve": curve,
        "best_saved_validation": best,
        "subset": json.loads((RAW_DIRECTORY / "subset.json").read_text()),
        "benchmark": json.loads(
            (logging_config.directory.parent / "benchmark.json").read_text()
        ),
        "torch_cpu_threads": torch.get_num_threads(),
        "seed": 42,
        "comparison": "Single seed, equal-token baseline; include preparation separately and rebenchmark each architecture for equal-time runs.",
    }
    tokenizer = ByteLevelBPETokenizer.load(
        preparation_config.output_directory / "tokenizer.json"
    )
    model, _ = build_model(tokenizer)
    best_path = checkpoint_config.directory / f"step-{best['update']:08d}.pt"
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    model.to("cuda").eval()
    samples = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = torch.tensor([tokenizer.encode(prompt)], device="cuda")
            for _ in range(MAX_NEW_TOKENS):
                predicted = model(ids[:, -chunking_config.context_length :])[
                    :, -1
                ].argmax(-1, keepdim=True)
                ids = torch.cat([ids, predicted], dim=1)
                if predicted.item() == tokenizer.token_to_id("<|eot|>"):
                    break
            samples.append(
                {"prompt": prompt, "text": tokenizer.decode(ids[0].tolist())}
            )
    report["greedy_samples_from_best_saved_checkpoint"] = samples
    report["best_checkpoint"] = str(
        best_path.relative_to(logging_config.directory.parent)
    )
    path = logging_config.directory.parent / "training-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
