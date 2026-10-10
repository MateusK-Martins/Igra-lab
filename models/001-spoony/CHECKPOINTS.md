# Training checkpoints and resume

`configs/training/checkpoint.py` selects the checkpoint directory and interval
(currently 100 successful optimizer updates). Pass that config and a settings
manifest to `train_loop(checkpoint_config=..., checkpoint_settings=...)` to
save periodically and after the final update. Files are named
`step-00000100.pt` under `results/pretraining/local/checkpoints/` by default.
Saving follows that update's validation and logging. Skipped FP16 updates do
not trigger saves. Checkpoints are written to a temporary file in the same
directory, then replaced atomically; failed writes preserve the previous file.

## What is restored

`src/training/checkpoint.py` saves model weights/mode, optimizer, scheduler,
scaler, training counters and latest metrics, CPU/CUDA RNG, both DataLoader
random generators, loader signatures, and the supplied settings. The training
state also records the batch cursor and generator state at the epoch's start.

For a partial epoch, resume reconstructs its shuffled order and skips consumed
batches, then restores the saved RNG states. This rereads consumed data but
preserves the next batch, dropout, and future epoch order. Checkpoints occur
between complete accumulation groups; pending gradients are not saved.

## Calling sequence

1. Rebuild the tokenizer, model, loaders, optimizer, scheduler, and precision
   components with the same settings. Move the model to its device before
   building the optimizer.
2. Call `load_checkpoint()` with those components and the settings manifest.
   It returns `TrainingState`; pass it as `state` to `train_loop()`.
3. Build the TensorBoard writer with `build_writer(logging_config,
   resume_step=state.updates)`. This hides events beyond the restored update
   if the previous process continued past that checkpoint.
4. Run the loop and close the writer in `finally`.

For example, with the already constructed components:

```python
state = load_checkpoint(
    checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    train_loader=train_loader,
    validation_loader=validation_loader,
    settings=settings,
)
writer = build_writer(logging_config, resume_step=state.updates)
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
        checkpoint_settings=settings,
    )
finally:
    writer.close()
```

For a fresh run, omit loading and the loop's `state` argument, and call
`build_writer(logging_config)` without `resume_step`.

## Settings and supported data

The manifest accepts JSON-compatible values, dataclass instances, and Paths.
The caller must include choices affecting model/data/updates: model definition,
optimizer, scheduler, precision, training, chunking and loader settings, plus
tokenizer/dataset hashes and Git revision. Saving does not discover these
identities. Loading compares the supplied manifest and loader signatures before
restoring components. The scheduler recipe must match because LambdaLR's
function is not stored in its state dictionary.

Exact continuation currently supports deterministic map-style datasets and
collation, `num_workers=0`, standard batching, explicit CPU loader generators,
and sequential sampling or shuffle without replacement. The CUDA device count
must match. Python/NumPy RNG, stochastic transforms, custom samplers, worker
prefetch, and distributed resume are outside this implementation. Exact tests
compare uninterrupted and resumed runs in the same environment; they do not
promise identical results across hardware or PyTorch versions.

`src/training/manifest.py` now provides `build_manifest()` and `save_manifest()`.
The builder checks both splits against the tokenizer's SHA-256, validates token
storage/document offsets, hashes all three files per split, and records the Git
commit plus Python source hashes under `src/`, `configs/model/`, and `scripts/`.
This captures uncommitted code edits too. Python, PyTorch, CUDA and GPU identities
are recorded. Hashing streams files rather than loading token stores into RAM.

In `scripts/train.py`, after constructing the components, build the manifest:

```python
from src.config.paths import project_root
from src.training.manifest import build_manifest, save_manifest

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
```

Use this complete dictionary as `settings` in `load_checkpoint()` and as
`checkpoint_settings` in `train_loop()`. After successful checkpoint loading
(or for a new run), call:

```python
save_manifest(manifest, logging_config.directory.parent / "manifest.json")
```

An identical existing manifest is kept; a different one raises without
replacing it. Select fresh output directories for a different run. Do not put
`run_config.resume_from` into settings: choosing a checkpoint is an invocation
action, not a change to the training recipe. The manifest contains no timestamps.
Exact comparison also requires the same Git commit and recorded environment;
even a documentation-only commit changes the manifest. The code and prepared
artifacts must stay unchanged while training. Source hashes record the model
recipe as code; they do not serialize the block definition objects.

`scripts/train.py` connects these helpers, optional checkpoint loading, writer
cleanup, and the loop. Set `configs/training/run.py` resume_from to a checkpoint
Path and run `.venv/bin/python -m scripts.train` from the model directory.
Automatic interruption saves, checkpoint retention, and
persistent-overflow stopping are not implemented.
