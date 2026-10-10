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

The runnable training script and complete run-manifest construction are the
next integration step. Automatic interruption saves, checkpoint retention, and
persistent-overflow stopping are not implemented.
