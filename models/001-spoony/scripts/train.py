import torch

from configs.data.dataloader import (
    chunking_config,
    train_loader_config,
    validation_loader_config,
)
from configs.model.model import TOKEN_MLP_VARIANT, build_model
from configs.tokenizers.preparation import preparation_config
from configs.training.checkpoint import checkpoint_config
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.run import run_config
from configs.training.scheduler import scheduler_config
from configs.training.training import logging_config, training_config
from src.config.paths import project_root
from src.data.dataloader import build_split_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.training.checkpoint import load_checkpoint
from src.training.logging import build_writer
from src.training.loop import train_loop
from src.training.manifest import build_manifest, save_manifest
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler


def main() -> None:
    device = torch.device(run_config.device)
    context, scaler = build_precision(device, precision_config)

    data_directory = preparation_config.output_directory
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

    model, _ = build_model(tokenizer)
    model = model.to(device)

    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)

    manifest = build_manifest(
        project_directory=project_root(),
        data_directory=data_directory,
        settings={
            "token_mlp_variant": TOKEN_MLP_VARIANT,
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


if __name__ == "__main__":
    main()
