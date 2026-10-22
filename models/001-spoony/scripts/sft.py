import torch

from configs.model.model import build_model
from configs.sft.data import data_config, train_loader_config, validation_loader_config
from configs.sft.optimizer import optimizer_config
from configs.sft.precision import precision_config
from configs.sft.run import initialization_config, run_config
from configs.sft.scheduler import scheduler_config
from configs.sft.training import checkpoint_config, logging_config, training_config
from configs.training.health import health_config
from src.config.paths import project_root
from src.data.dataloader import build_dataloader
from src.data.tokenizer import ByteLevelBPETokenizer
from src.sft.data import SFTDataset, collate_instructions
from src.training.checkpoint import load_checkpoint, load_model_weights
from src.training.health import HealthMonitor
from src.training.logging import build_writer
from src.training.loop import train_loop
from src.training.manifest import build_artifact_manifest, file_sha256, save_manifest
from src.training.optimizer import build_optimizer
from src.training.precision import build_precision
from src.training.scheduler import build_scheduler


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if run_config.resume_from is None and any(
        checkpoint_config.directory.glob("step-*.pt")
    ):
        raise FileExistsError("Existing checkpoints; configure resume_from")
    device = torch.device(run_config.device)
    context, scaler = build_precision(device, precision_config)

    tokenizer = ByteLevelBPETokenizer.load(data_config.tokenizer_path)
    train_loader = build_dataloader(
        SFTDataset(data_config.train_path, tokenizer, data_config),
        train_loader_config,
        collate_fn=collate_instructions,
    )
    validation_loader = build_dataloader(
        SFTDataset(data_config.validation_path, tokenizer, data_config),
        validation_loader_config,
        collate_fn=collate_instructions,
    )

    model, _ = build_model(tokenizer)
    if run_config.resume_from is None:
        load_model_weights(
            initialization_config.checkpoint_path,
            model=model,
            tokenizer_path=data_config.tokenizer_path,
        )
    model = model.to(device)

    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)

    manifest = build_artifact_manifest(
        project_directory=project_root(),
        artifacts={
            "tokenizer.json": file_sha256(data_config.tokenizer_path),
            "train.jsonl": file_sha256(data_config.train_path),
            "validation.jsonl": file_sha256(data_config.validation_path),
        },
        settings={
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
            "precision": precision_config,
            "training": training_config,
            "stage": "sft",
            "data": data_config,
            "initialization": initialization_config,
            "train_loader": train_loader_config,
            "validation_loader": validation_loader_config,
            "device": run_config.device,
        },
    )

    if run_config.resume_from is not None:
        saved = torch.load(
            run_config.resume_from, map_location="cpu", weights_only=True
        )
        manifest["artifacts"]["initial_checkpoint"] = saved["settings"]["artifacts"][
            "initial_checkpoint"
        ]

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

    if run_config.resume_from is None:
        manifest["artifacts"]["initial_checkpoint"] = file_sha256(
            initialization_config.checkpoint_path
        )

    save_manifest(
        manifest,
        logging_config.directory.parent / "manifest.json",
    )

    writer = build_writer(
        logging_config,
        resume_step=None if state is None else state.updates,
    )

    with HealthMonitor(logging_config.directory.parent, health_config) as monitor:
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
                stop_requested=lambda: monitor.reason,
            )
        finally:
            writer.close()
    if state.updates < scheduler_config.total_steps:
        print("SFT stopped safely", flush=True)
        return
    from src.training.export import export_model

    export_model(
        logging_config.directory.parent / "model",
        model=model,
        tokenizer_path=data_config.tokenizer_path,
        manifest=manifest,
        state=state,
    )
    print("SFT complete", flush=True)


if __name__ == "__main__":
    main()
