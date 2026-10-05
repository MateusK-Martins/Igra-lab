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

No real corpus is selected: `configs/data/datasets.py` defines an empty pipe.
GQA attention now computes output using injected positional rotation, explicit
causal/window masks, SDPA, and the output projection. RoPE is a configurable
function bound with `functools.partial`. Cached calls write new rotated K/V
before reading retained history. Bounded chunks can discard context for earlier
queries; see MODEL_ASSEMBLY.md for that behavior.
Mixed precision, the training loop, checkpoint orchestration, evaluation, and
SFT remain pending. Infrastructure is being built before choosing a corpus or
starting training. The latest suite passed 323 tests.

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
saved in its state. Full training/checkpoint integration is still pending.

## Directory contract

- `src/model/` defines block contracts, construction, and the runtime model.
- `src/config/types.py` defines the typed configuration contracts.
- `configs/data/` holds dataset definitions.
- `configs/data/dataloader.py` holds window and train/validation batch settings.
- `configs/tokenizers/` holds tokenizer preparation settings.
- `configs/model/model.py` exposes `build_model(tokenizer)`.
- `configs/training/` holds optimizer and scheduler settings.
- `src/data/`, `src/training/`, `src/evaluation/`, and `src/inference/` hold
  separate stages of the model lifecycle.
- `tests/` contains tests that run without a full training job.
- `results/` stores small, versioned experiment records only.
