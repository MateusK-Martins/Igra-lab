import json

import torch
from torch.utils.data import ConcatDataset

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.data.qa_corpus import qa_config
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.run import run_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config, training_config
from src.config.paths import project_root
from src.data.dataloader import build_dataloader, build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.sft.data import SFTDataset
from src.training.checkpoint import load_checkpoint
from src.training.logging import build_writer
from src.training.loop import train_loop
from src.training.manifest import build_manifest, file_sha256, save_manifest
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler

expected_artifacts_path = project_root() / "configs/data/wikipedia-artifacts.json"


def ensure_prepared_data(data_directory):
    if not data_directory.exists():
        from scripts.prepare_wikipedia import main as prepare

        print("Preparing pinned Wikipedia data before training", flush=True)
        prepare()
    expected = json.loads(expected_artifacts_path.read_text())
    for relative, digest in expected.items():
        path = data_directory / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(
                f"Prepared data does not match pinned Wikipedia artifact: {relative}"
            )


def main() -> None:
    device = torch.device(run_config.device)
    context, scaler = build_precision(device, precision_config)

    if run_config.resume_from is None and any(
        checkpoint_config.directory.glob("step-*.pt")
    ):
        raise FileExistsError(
            "Checkpoints already exist. Use --resume-latest or choose a new run directory."
        )
    torch.set_num_threads(4)
    data_directory = preparation_config.output_directory
    ensure_prepared_data(data_directory)
    tokenizer = ByteLevelBPETokenizer.load(data_directory / "tokenizer.json")

    train_loader = build_split_dataloader(
        data_directory / "train",
        chunking_config,
        train_loader_config,
    )
    validation_loader = build_split_dataloader(
        data_directory / "validation",
        chunking_config,
        validation_loader_config,
    )

    if qa_config is not None:
        train_loader = build_dataloader(
            ConcatDataset(
                [
                    train_loader.dataset,
                    SFTDataset(qa_config.train_path, tokenizer, qa_config),
                ]
            ),
            train_loader_config,
        )

    model, _ = build_model(tokenizer)
    model = model.to(device)

    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)

    manifest = build_manifest(
        project_directory=project_root(),
        data_directory=data_directory,
        settings={
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
            "precision": precision_config,
            "training": training_config,
            "chunking": chunking_config,
            "train_loader": train_loader_config,
            "validation_loader": validation_loader_config,
            "device": run_config.device,
        },
    )

    if qa_config is not None:
        manifest["settings"]["qa_format"] = {
            "instruction_prefix": qa_config.instruction_prefix,
            "response_prefix": qa_config.response_prefix,
            "end_token": qa_config.end_token,
            "context_length": qa_config.context_length,
        }
        manifest["artifacts"]["qa-train.jsonl"] = file_sha256(qa_config.train_path)
        manifest["artifacts"]["qa-validation.jsonl"] = file_sha256(
            qa_config.validation_path
        )

    state = None

    if run_config.resume_from is not None:
        state = load_checkpoint(
            run_config.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            validation_loader=validation_loader,
            settings=manifest,
        )

    save_manifest(
        manifest,
        logging_config.directory.parent / "manifest.json",
    )

    writer = build_writer(
        logging_config,
        resume_step=None if state is None else state.updates,
    )

    try:
        state = train_loop(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
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

    from src.training.export import export_model

    export_model(
        logging_config.directory.parent / "model",
        model=model,
        tokenizer_path=data_directory / "tokenizer.json",
        manifest=manifest,
        state=state,
    )
    print(
        f"Training complete. Model: {logging_config.directory.parent / 'model'}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse
    from dataclasses import replace
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Prepare pinned Wikipedia data and train the configured model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--resume-latest", action="store_true")
    arguments = parser.parse_args()
    resume = arguments.resume
    if arguments.resume_latest:
        candidates = sorted(checkpoint_config.directory.glob("step-*.pt"))
        if not candidates:
            parser.error("No checkpoints exist in the configured checkpoint directory")
        resume = candidates[-1]
    if resume is not None:
        run_config = replace(run_config, resume_from=resume)
    main()
