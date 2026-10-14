"""Train a fixed story/QA mixture, then report each held-out task separately."""

import json
import subprocess
import time
from dataclasses import asdict

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.data.qa_corpus import qa_config
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.comparison import comparison_config
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.run import run_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config, training_config
from scripts.comparison_common import loaders
from src.config.paths import project_root
from src.evaluation.qa import evaluate_answers
from src.training.checkpoint import load_checkpoint
from src.training.logging import build_writer
from src.training.loop import train_loop
from src.training.manifest import build_artifact_manifest, file_sha256, save_manifest
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler
from src.training.state import TrainingState
from src.training.validation import validate


def main():
    torch.set_num_threads(comparison_config.cpu_threads)
    tokenizer, train, stories, qa = loaders()
    device = torch.device(run_config.device)
    model, entries = build_model(tokenizer)
    model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    context, scaler = build_precision(device, precision_config)
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)
    data = preparation_config.output_directory
    artifacts = {
        path.relative_to(data).as_posix(): file_sha256(path)
        for path in data.rglob("*")
        if path.is_file()
    }
    manifest = build_artifact_manifest(
        project_directory=project_root(),
        artifacts=artifacts,
        settings={
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
            "precision": precision_config,
            "training": training_config,
            "chunking": chunking_config,
            "train_loader": train_loader_config,
            "validation_loader": validation_loader_config,
            "comparison": comparison_config,
            "qa_format": {
                "instruction_prefix": qa_config.instruction_prefix,
                "response_prefix": qa_config.response_prefix,
                "end_token": qa_config.end_token,
                "context_length": qa_config.context_length,
            },
            "mixture": "three story batches then one response-only QA batch; deterministic pre-shuffled indexes",
            "device": run_config.device,
        },
    )
    root = logging_config.directory.parent
    state = TrainingState()
    if run_config.resume_from is not None:
        state = load_checkpoint(
            run_config.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train,
            validation_loader=stories,
            settings=manifest,
        )
    save_manifest(manifest, root / "manifest.json")
    writer = build_writer(
        logging_config, resume_step=None if state.updates == 0 else state.updates
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        state = train_loop(
            model=model,
            train_loader=train,
            validation_loader=stories,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            scaler=scaler,
            device=device,
            config=training_config,
            total_steps=scheduler_config.total_steps,
            writer=writer,
            logging_config=logging_config,
            state=state,
            checkpoint_config=checkpoint_config,
            checkpoint_settings=manifest,
        )
    finally:
        writer.close()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    evaluation_start = time.perf_counter()
    story_loss, story_perplexity = validate(model, stories, context, device)
    qa_loss, qa_perplexity = validate(model, qa, context, device)
    answers = evaluate_answers(
        model, entries, tokenizer, qa_config, comparison_config, context, device
    )
    (root / "qa-predictions.json").write_text(
        json.dumps(answers.pop("predictions"), indent=2) + "\n"
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    events = EventAccumulator(str(logging_config.directory)).Reload()
    result = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "recipe_commit": manifest["code"]["git_commit"],
        "parameters": parameters,
        "vocab_size": tokenizer.vocab_size,
        "training_seconds_including_validation_and_saving": seconds,
        "evaluation_seconds": evaluation_seconds,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_vram_bytes": peak,
        "training_state": {
            key: value
            for key, value in asdict(state).items()
            if not isinstance(value, torch.Tensor)
        },
        "story_validation": {
            "documents": 1000,
            "windows": len(stories.dataset),
            "loss": story_loss,
            "perplexity": story_perplexity,
        },
        "qa_validation": {
            "examples": len(qa.dataset),
            "answer_only_loss": qa_loss,
            "answer_only_perplexity": qa_perplexity,
            "loss_includes_end_token": True,
            "loss_uses_first_reference": True,
            "generation": answers,
            "subset": "first length-eligible official validation records; full context retained",
        },
        "story_validation_curve": [
            {"update": event.step, "loss": event.value}
            for event in events.Scalars("validation/loss")
        ],
        "preparation": json.loads((data / "preparation-report.json").read_text()),
        "benchmark": json.loads((root / "benchmark.json").read_text()),
        "checkpoint": f"checkpoints/step-{state.updates:08d}.pt",
        "comparison_protocol": "Final-budget models, one seed, common tokenizer/artifacts/batch order/settings. QA EM/F1 is a filtered validation subset, not full SQuAD leaderboard scoring.",
    }
    (root / "training-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("preparation", "story_validation_curve")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
