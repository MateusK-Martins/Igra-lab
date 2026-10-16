> **Active:** [LATENT_GMLP_14M.md](LATENT_GMLP_14M.md), gMLP after16D attention retrieval, before384D output projection. Prior notes below are historical.

> **Active:** [GMLP_TIDE_A16.md](GMLP_TIDE_A16.md), `[Tide,gMLP,A16/RoPE] ×2`,64-token fixed gMLP windows. Notes below are historical.

> **Active:** [TIDE_A16.md](TIDE_A16.md), four Tide / two A16+RoPE mixers. Notes below are historical.

# Tide/Lantern hybrid14M

Current experiment: [HYBRID_14M.md](HYBRID_14M.md).
The material below is inherited historical context.

# Tide

Tide is the continuously ranged-initialized No-K model with independent write gates and global RMSNorm. It has 13,774,080 parameters and 384 FP32 recurrent state values per layer.

The frozen training recipe is commit 744e1ca; completed results are at 563ab4a. Validation loss: **2.010153**; perplexity: **7.464462**.

The following experiment notes and original Spoony documentation are retained as history.

Current initialization controls: see [INITIALIZATION_CONTROLS.md](INITIALIZATION_CONTROLS.md).

Current mechanism ablations: see [MECHANISM_ABLATIONS.md](MECHANISM_ABLATIONS.md).

Current No-K gate/readout suite: see [GATE_READOUT_TESTS.md](GATE_READOUT_TESTS.md).

Current controlled mechanism suite: see [MECHANISMS.md](MECHANISMS.md). Earlier notes are historical.

> **Active token-decay experiment:** see [TOKEN_DECAY.md](TOKEN_DECAY.md).
> Ember and earlier notes below are historical.

> **Active Ember experiments:** read [DECAY_EXPERIMENTS.md](DECAY_EXPERIMENTS.md).
> RMSNorm is restored; the next runs add parallel feature-wise state decay.

> **Active attentionless branch:** read [ATTENTIONLESS.md](ATTENTIONLESS.md)
> for the cumulative mechanism, cache semantics and current experiment.
> Lantern notes below are historical baseline results.

> **Codename branch:** the active larger TinyStories recipe and comparison
> protocol are documented in [CODENAME.md](CODENAME.md). The historical
> Spoony notes below describe earlier configurations.

# Spoony

Spoony is model 001 in Igra-lab: a small, independent decoder-only Transformer
used as the first research baseline.

Its purpose is to provide a fast, understandable local training loop on the
RTX 4050 while establishing contracts that later decoder Transformer models can
reuse through the template branch.

## Environment

Each model owns its virtual environment. Create it in the model directory and
install the dependency set that matches the machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/local-cu128.txt
```

`requirements/base.txt` lists packages used on every machine. The local CUDA
file adds the PyTorch build tested on the RTX 4050. Cloud dependency files are
added only after selecting the cloud GPU and its compatible CUDA build.

## Status

Implemented and tested:

- Byte-level BPE training, persistence, and a generic tokenizer protocol.
- Source, DataSet, and DataSetPipe contracts with one dataset per split.
- Train-only tokenizer training, document encoding, and end-of-document tokens.
- Binary token storage, document offsets, and tokenizer hashes.
- Disk-backed next-token windows and seeded PyTorch DataLoaders.
- Declarative model validation and assembly with Sequential, Repeat, residuals,
  embeddings, Linear, SwiGLU, RMSNorm, and an optionally tied LM head.
- Expanded leaf assembly into a ModuleList, per-occurrence IDs, and external
  cache-entry collection through `definition.assemble()`.
- Bounded/unbounded KV storage, device transfers, snapshot save/load, and
  optional cache forwarding through runtime blocks.
- A four-layer decoder configured through `configs/model/model.py`, with a
  supplied trained tokenizer, width 256, windowed GQA, RoPE, and a tied LM head.
- Next-token cross-entropy loss, configurable AdamW construction, and validated
  warmup/cosine scheduler settings with state-restoration tests.
- FP32/BF16/FP16 contexts, FP16 scaling, accumulation, clipping, and a training
  loop with update/batch/token/epoch counters.
- Standalone token-weighted validation loss and perplexity with mode restoration.

No real corpus is selected: `configs/data/datasets.py` defines an empty pipe.
GQA attention now computes output using injected positional rotation, explicit
causal/window masks, SDPA, and the output projection. RoPE is a configurable
function bound with `functools.partial`. Cached calls write new rotated K/V
before reading retained history. Bounded chunks can discard context for earlier
queries; see MODEL_ASSEMBLY.md for that behavior.
The runnable training script connects the components, with tested fresh-run and
resume paths. Generation and SFT remain pending. No real corpus training has
been performed. The latest suite passed 450 tests.

See [ROADMAP.md](ROADMAP.md) and [MODEL_ASSEMBLY.md](MODEL_ASSEMBLY.md) for the
remaining work and composition contracts.

## Commands

Run from this model directory with its virtual environment activated:

```bash
python -m pytest -q
ruff check src configs scripts tests
ruff format --check src configs scripts tests
python -m scripts.prepare_data
```

Preparation requires train datasets to be defined first. It trains BPE, saves
the tokenizer, and processes train and validation. Existing split artifacts
are rejected rather than overwritten; select a fresh output directory for a
new preparation.

Importing `scripts` calls `torch.manual_seed(42)`. DataLoaders retain their own
configured generators. This does not promise deterministic CUDA execution or
seed Python/NumPy; there is no separate seed module or custom weight initializer.

## Model and training components

Load the tokenizer saved by data preparation, then build the configured model:

```python
from configs.model.model import build_model
from configs.tokenizers.preparation import preparation_config
from src.data.tokenizer import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer.load(
    preparation_config.output_directory / "tokenizer.json"
)
model, cache_entries = build_model(tokenizer)
```

Each of four layers applies residual attention, RMSNorm, residual SwiGLU, and
RMSNorm. GQA has eight query heads, two KV heads, head width 32, window 128,
dropout zero, and RoPE base 10000. SwiGLU hidden width is 688. With vocabulary
4096, the model has 3,819,520 unique parameters; actual vocabulary size comes
from the loaded tokenizer. PyTorch's default weight initialization is retained.

`next_token_loss(logits, targets)` consumes `[B, T, V]` logits and `[B, T]`
targets already shifted by the dataset. It returns mean cross-entropy and does
not shift targets again. `build_optimizer(model, optimizer_config)` constructs
AdamW with one parameter group, applying the declared decay to all parameters;
the tied embedding/head parameter appears once.

`build_scheduler(optimizer, scheduler_config)` constructs LambdaLR and sets the
first update's learning rate immediately. Call `scheduler.step()` after each
successful `optimizer.step()`, not after every accumulation batch. Warmup uses
factors `1/warmup_steps` through 1; cosine reaches `min_lr_ratio` on the final
planned update and stays there afterwards. Without warmup, the first update
uses the base learning rate, including the single-update case.

Current editable values are AdamW LR 3e-4, betas (0.9, 0.95), epsilon 1e-8,
decay 0.01, and a schedule of 100 warmup updates, 1000 total updates, and minimum
LR ratio 0.1. These are provisional settings, not measured training results.
For resume, rebuild optimizer and scheduler with the same configuration before
loading both state dictionaries; the LambdaLR closure's configuration is not
saved in its state. Checkpoint loading checks the supplied run settings before
restoration.

## Directory contract

- `src/model/` defines block contracts, construction, and the runtime model.
- `src/config/types.py` defines the typed configuration contracts.
- `configs/data/` holds dataset definitions.
- `configs/data/dataloader.py` holds window and train/validation batch settings.
- `configs/tokenizers/` holds tokenizer preparation settings.
- `configs/model/model.py` exposes `build_model(tokenizer)`.
- `configs/training/` holds optimizer, scheduler, precision, training, logging,
  and checkpoint settings.
- `src/data/`, `src/training/`, `src/evaluation/`, and `src/inference/` hold
  separate stages of the model lifecycle.
- `tests/` contains tests that run without a full training job.
- `results/` stores small, versioned experiment records only.

## Training execution

`build_precision(device, config)` returns a reusable autocast/null context and
a GradScaler enabled only for CUDA FP16. BF16 is the current editable mode.
`train_step()` accumulates token-weighted losses, unscales before clipping, and
steps the scheduler only after an actual optimizer update. FP16 overflow tests
verify skipped updates and subsequent recovery.

`TrainingConfig` declares accumulation_steps, max_grad_norm, and
validation_interval (currently 1, 1.0, and 100). `train_loop()` groups DataLoader
batches, processes partial groups, restarts
epochs, and returns TrainingState. Counters track successful updates, consumed
batches/tokens, completed epochs, and the last training loss. The model must be
on the target device before constructing the optimizer. Pass the scheduler
configuration's total_steps as the loop limit.

`validate()` uses eval mode and no_grad, computes token-weighted mean loss and
perplexity, and restores the previous model mode, including on failure. It does
not use a cache or optimizer. The loop receives a separate validation_loader
and validates after each configured interval of successful updates and after
the final update. Counters advance before the interval check, and skipped
updates do not repeat validation. TrainingState stores the latest validation
loss and perplexity. TensorBoard records training loss, the LR used for the update,
consumed tokens, validation loss, and perplexity. Logging settings live in
`configs/training/training.py`; checkpoint settings live in
`configs/training/checkpoint.py`. See [CHECKPOINTS.md](CHECKPOINTS.md) for saving
and resuming. `scripts/train.py` connects these components.


## Local training workflow

1. Define train and validation sources in `configs/data/datasets.py`, and select
   the preparation directory/tokenizer settings in `configs/tokenizers/preparation.py`.
   Run `.venv/bin/python -m scripts.prepare_data` once. It trains BPE on train
   documents and writes tokenizer.json plus both processed splits. Existing
   preparation artifacts are rejected instead of overwritten.
2. Review `configs/model/model.py`, `configs/data/dataloader.py`, and
   `configs/training/`. `run.py` selects CPU/CUDA and an optional checkpoint;
   precision, optimizer, scheduler, accumulation, validation, logging, and
   checkpoint settings live in their existing files. Use matching fresh logging
   and checkpoint directories for a new experiment.
3. Run `.venv/bin/python -m scripts.train` from the model directory. The script
   loads the saved tokenizer and splits, builds precision/model/optimizer/scheduler,
   and computes the manifest. The model is moved before optimizer construction.
   Training uses no KV cache.
4. The script loads a checkpoint if `run_config.resume_from` is set, using the
   manifest to check compatibility. It then saves/compares `manifest.json` in
   the logging directory's parent, creates the TensorBoard writer, and calls
   `train_loop()` until `scheduler_config.total_steps` successful updates.
   The writer is closed in `finally`, including after training errors.
5. Inspect TensorBoard with
   `.venv/bin/tensorboard --logdir results/pretraining/local/tensorboard`.
   Checkpoints are under
   `results/pretraining/local/checkpoints/` by default. Validation, logging, and
   checkpoints follow their configured update intervals and the final update.
6. To resume, set `resume_from` to a checkpoint Path and rerun the same command.
   Keep the recipe, prepared artifacts, recorded code/Git revision/environment,
   and output directories compatible. The script restores training/RNG/data
   state and hides TensorBoard events beyond the restored step. `resume_from`
   itself is excluded from the manifest settings.

No prepared real corpus is configured yet. The next verification step is a
controlled one-batch overfit experiment, followed by corpus selection/preparation
and a short smoke run. Integration tests currently use a tiny model with real
prepared artifacts and the real loop; they do not establish convergence of the
configured four-layer model.
