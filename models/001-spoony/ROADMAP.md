# Spoony roadmap

Spoony is a small, independently runnable decoder Transformer. Its size is
chosen for fast local training, but its data, training, evaluation, cloud, and
supervised fine-tuning systems must use scalable research-quality contracts.

TinyStories is a candidate for the first pretraining experiment. No corpus is
configured yet; data contracts and preparation have been tested with fake
sources before choosing a real source.

## Current progress

Data source composition, BPE preparation, token storage, disk-backed windows,
DataLoaders, and declarative assembly are implemented. Model composition
includes embeddings, Linear, SwiGLU, RMSNorm, expanded Sequential/Repeat
definitions, leaf residuals, and head tying. Assembly returns a flat ModuleList
runtime and cache entries. External KV storage, snapshots, IDs, and optional
cache forwarding are implemented. The latest verification passed 323 tests.

GQA, injected RoPE, causal/window masks, and cached execution are implemented.
The editable four-layer architecture, next-token loss, AdamW builder, and
warmup/cosine scheduler are implemented. Next: mixed precision and the training
step. Build the infrastructure before selecting a corpus or starting training.
Still missing: a configured corpus, training loop, checkpoint orchestration,
evaluation, generation, and SFT. The following lifecycle sections describe
targets unless explicitly identified as implemented.

## Project rules

- Every experiment choice belongs in explicit Python dataclass configuration
  under `configs/`. Definition classes and implementation code stay under
  `src/`. There is no configuration parser or generic merge framework.
- Each model directory remains self-contained.
- Large datasets, checkpoints, local environments, event logs, and secrets are
  never committed.
- Small essential artifacts, such as tokenizer JSON files, tokenizer metadata,
  configuration snapshots, reports, and model cards are committed or attached
  to releases.
- Every meaningful run records its Git commit, resolved configuration, dataset
  manifest, tokenizer hash, hardware, and metrics.

## Target layout

```text
models/001-spoony/
├── configs/
│   ├── data/
│   ├── tokenizers/
│   ├── model/           # Implemented build_model(tokenizer)
│   └── training/        # Optimizer and scheduler settings
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── results/
│   ├── pretraining/
│   └── sft/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── model/
│   ├── sft/
│   └── training/
├── tests/
├── requirements/
├── README.md
├── ROADMAP.md
└── pyproject.toml
```

`data/raw/`, `data/processed/`, `data/cache/`, `checkpoints/`, `.venv/`, real
`.env` files, and local tracking logs are ignored by Git.

## Configuration

Configuration is Python using dataclasses. The dataclasses are typed contracts;
editable modules instantiate them:

```text
src/config/
├── types.py       # Dataclasses and their validation rules
└── snapshot.py    # Save a resolved configuration as JSON

configs/
├── data/
│   ├── datasets.py    # Selected sources and manifests
│   └── dataloader.py  # Chunking and batch settings
├── tokenizers/
│   └── preparation.py # Tokenizer settings and output directory
├── model/
│   └── model.py       # Architecture factory receiving a trained tokenizer
└── training/
    ├── optimizer.py   # AdamW settings
    └── scheduler.py   # Warmup/cosine settings
```

`dataclasses.replace` can create variations without a custom merge language.
Constructors validate settings. A JSON snapshot helper exists for JSON-compatible
dataclass fields; Path-containing and model definitions need a serialization
policy before full run snapshots are integrated. Saving resolved settings in
run directories and checkpoints remains a training-system requirement.

Local and cloud training will differ through editable settings while sharing
the same training implementation.

## Data system

Source modules:

```text
src/data/
├── schemas.py
├── sources.py
├── pipeline.py
├── tokenizer.py
├── prepare.py
├── serialization.py
├── chunker.py
└── dataloader.py
```

Responsibilities:

- `schemas.py`: validated document and metadata structures.
- `sources.py`: download or stream a declared source.
- `pipeline.py`: DataSet binds a DocumentManifest to a Source; DataSetPipe
  traverses selected splits in declared dataset order.
- `tokenizer.py`: train, save, load, encode, and decode BPE.
- `prepare.py`: train BPE from train documents only, then encode each selected
  split with a trained tokenizer and write artifacts through TokenStore.
- `serialization.py`: efficient local read and write of prepared token data.
- `chunker.py`: next-token windows from a token stream.
- `dataloader.py`: batches, shuffling, workers, and pinned memory.

Dataset definitions live in `configs/data/datasets.py`. Each DataSet represents
one internal split. DocumentManifest declares name, language, text field,
split, license, and citation. HuggingFaceSource owns the external dataset ID,
revision, source split, and streaming setting. A source must provide a fresh,
stable traversal each time read() is called.

The source adapter is generic, not TinyStories-specific. For a corpus without
supplied validation data, deterministic document splitting must be added before
training the tokenizer. That splitting mechanism is not implemented yet.

## Tokenizer

Spoony uses a byte-level BPE tokenizer. It handles UTF-8 text while learning
frequent merges from training data.

```text
training documents
    ↓
train BPE only on training documents
    ↓
save tokenizer.json and metadata
    ↓
encode training and validation documents with that tokenizer
```

Never use validation documents to train BPE.

The tokenizer artifact is stored under:

```text
data/processed/<preparation-id>/
├── tokenizer.json
├── train/
│   ├── tokens.bin
│   ├── documents.jsonl
│   └── metadata.json
└── validation/
    ├── tokens.bin
    ├── documents.jsonl
    └── metadata.json
```

Tokens are little-endian uint32 IDs. Each document ends with the configured
end token; JSONL offsets preserve its interval with an exclusive end. Split
metadata records format version, dtype, document count, token count, and the
tokenizer file's SHA-256. DataProcessor verifies the same tokenizer is used for
both splits. Additional source/configuration provenance is future work.

The model uses the tokenizer's actual vocabulary size. Prepared artifacts are
ignored by Git; small selected tokenizer artifacts may later be attached to
releases.

## Transformer model

Target modules:

```text
src/model/
├── definitions.py      # Implemented block protocols and dataclasses
├── assembler.py        # Implemented Model validation and assemble()
├── composition.py      # Implemented residual wrapping and build_block()
├── language_model.py   # Implemented embedding -> body -> logits
├── feedforward.py      # Implemented SwiGLU and cache-compatible Linear
├── position.py         # PositionRotation protocol and RoPE function
├── stability.py        # Implemented RMSNorm
├── attention.py        # GQA, RoPE injection, masks, SDPA, cache writes/reads
└── cache_storage.py    # Implemented KV storage and snapshots
```

The configured architecture is a decoder-only Transformer:

```text
Token embeddings
    ↓
Four Transformer blocks
    ├── causal GQA with RoPE on Q/K + residual
    ├── RMSNorm
    ├── SwiGLU + residual
    └── RMSNorm
    ↓
Vocabulary projection tied to token embeddings
```

`configs/model/model.py` exposes `build_model(tokenizer)` with width 256,
8 query heads, 2 KV heads, head width 32, attention window 128, dropout zero,
RoPE base 10000, SwiGLU hidden width 688, and RMSNorm epsilon 1e-6. At vocabulary
4096, it has 3,819,520 unique parameters. Default PyTorch initialization is used.
Model includes the supplied trained tokenizer,
embedding, body blocks, and LM head. Definitions own build(); the assembler
expands groups before validating adjacent dimensions and wraps leaf residuals.
Repeat builds fresh parameters per occurrence; group residuals are not applied.
Model.assemble() returns (ModelImpl, cache_entries). See MODEL_ASSEMBLY.md.

Use GQA from the outset with PyTorch scaled dot-product attention. Current
definitions declare num_query_heads, num_kv_heads, head_features, and window_size.
RoPE rotates Q/K inside attention through the PositionRotation callable. Cache
capacity and attention visibility are separate. Execution writes new K/V before
reading history; bounded chunks may lose earlier queries' context. See the assembly
guide for the tested semantics.

## Pretraining

Target modules:

```text
src/training/
├── state.py
├── loss.py         # Implemented mean next-token cross-entropy
├── optimizer.py    # Implemented configurable AdamW builder
├── scheduler.py    # Implemented warmup/cosine LambdaLR
├── precision.py    # Next: FP32/BF16/FP16 and scaling
├── distributed.py
├── checkpoint.py
├── logging.py
├── loop.py
└── train.py
```

Implemented components:

- `scripts/__init__.py` sets `torch.manual_seed(42)`; DataLoaders use their
  configured generators. There is no separate seed module.
- Loss reshapes `[B, T, V]` logits and already shifted `[B, T]` targets.
- Validated AdamW settings: positive finite LR/epsilon, nonnegative finite
  decay, and two finite betas in `[0, 1)`. One group applies decay to all weights.
- Validated scheduler settings: positive total updates, warmup in
  `[0, total_steps)`, and finite minimum LR ratio in `[0, 1]`.
- Scheduler construction sets the first LR; warmup rises linearly, cosine
  reaches the floor on the final planned update, and further steps stay there.
  Without warmup, the first update uses the base LR. Resume tests rebuild with
  identical settings and load both optimizer/scheduler state dictionaries.

Still required:

- Gradient clipping and gradient accumulation.
- Configurable FP16 or BF16 mixed precision.
- Gradient scaler when FP16 is selected.
- TensorBoard metrics.
- Periodic validation loss, perplexity, and fixed-prompt generation.
- Checkpoint creation and exact resume.
- RNG state capture/restoration for reproducible checkpoint continuation.
- One-GPU local mode and `torchrun` distributed mode.

A checkpoint contains model, optimizer, scheduler, scaler, step, consumed
tokens, random-number-generator state, resolved configuration, tokenizer hash,
dataset hash, and Git commit.

## Evaluation and generation

```text
src/evaluation/
├── perplexity.py
├── generation.py
├── prompts.py
├── benchmark.py
└── report.py
```

Generation uses configuration for seed, temperature, top-k, top-p, repetition
penalty, and output length. Save fixed prompts and generated samples to make
checkpoint comparisons meaningful.

For TinyStories, keep a stable set of story prompts across every run.

## Supervised fine-tuning

SFT begins only after a usable base checkpoint exists.

```text
src/sft/
├── schema.py
├── chat_template.py
├── prepare.py
├── dataset.py
├── masking.py
├── train.py
└── evaluate.py
```

The canonical source record is JSONL with messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are Spoony."},
    {"role": "user", "content": "Write a story about a fox."},
    {"role": "assistant", "content": "Once upon a time..."}
  ]
}
```

The SFT pipeline validates the schema, renders a versioned chat template,
tokenizes the conversation, and constructs labels where only assistant tokens
contribute to loss:

```text
system tokens     → ignore label
user tokens       → ignore label
assistant tokens  → token ID label
```

It must also support configured truncation or packing, end-of-turn tokens, and
held-out SFT evaluation prompts.

## Scripts

Scripts are thin command-line entry points. Testable logic stays under `src/`.

```text
scripts/
├── prepare_data.py
├── train_pretrain.py
├── evaluate.py
├── generate.py
├── train_sft.py
├── export_checkpoint.py
└── inspect_run.py
```

Each script imports its configuration explicitly from the relevant folder
under `configs/`. Dataset definitions live in `configs/data/` and tokenizer
preparation settings live in `configs/tokenizers/`. Importing `scripts` applies
the PyTorch seed. Only `prepare_data.py` currently exists as a runnable script.

## Tests

```text
tests/
├── test_config.py
├── test_tokenizer.py
├── test_data_prepare.py
├── test_chunker.py
├── test_dataloader.py
├── test_attention.py
├── test_model_shapes.py
├── test_causal_mask.py
├── test_loss.py
├── test_training_smoke.py
├── test_checkpoint_resume.py
└── test_sft_masking.py
```

Important checks:

- BPE save/load preserves encoding.
- Dataset splitting is deterministic.
- Chunker targets are shifted by one token.
- Model input `[B, T]` produces logits `[B, T, V]`.
- Causal attention cannot access future tokens.
- A tiny model overfits one batch.
- Resumed training reproduces the next step.
- SFT loss includes assistant tokens only.

CPU tests should be suitable for CI. GPU and distributed smoke tests run
locally or in cloud environments.

## Results

```text
results/
├── pretraining/
│   └── run-<id>/
│       ├── summary.json
│       ├── resolved-config.json
│       ├── samples.md
│       └── charts/
└── sft/
    └── run-<id>/
        ├── summary.json
        ├── resolved-config.json
        └── evaluation.md
```

Commit small reports, charts, and generated samples. Do not commit weights,
raw data, processed token arrays, or TensorBoard event files.

## Cloud scaling

The planned training code will support local and cloud runs:

```text
Local RTX 4050       Cloud GPU
---------------      ----------------------
one GPU              one or many GPUs
small batch          larger global batch
gradient accumulation torchrun and DDP
fast tests           long benchmark runs
```

Later performance work, only after correctness is established:

- PyTorch scaled dot-product attention and Flash Attention paths.
- `torch.compile`.
- Gradient and activation checkpointing.
- FSDP for models too large for one device.
- Container or cloud setup scripts.

## Git workflow

```text
template/base
    ↓
template/decoder-transformer
    ↓
model/001-spoony
    ↓
develop
    ↓
release/001-spoony-v0.1.0
    ↓
main
```

Model-specific code remains in Spoony. A reusable improvement is tested in
Spoony and then intentionally promoted to `template/decoder-transformer`.

## Build order

1. Completed: typed dataclass contracts and editable configuration folders.
2. Completed: generic sources, per-split DataSet, and DataSetPipe.
3. Completed: train-only BPE training, preparation, and token serialization.
4. Completed: disk-backed windows and configured DataLoader construction.
5. Completed: expanded assembly, embeddings, head tying, SwiGLU, RMSNorm,
   KV storage/snapshots, per-layer IDs, and external cache forwarding.
6. Completed: injected RoPE, causal/windowed GQA, SDPA, and cache writes/reads.
7. Completed: attention reference/gradient, causal isolation, window, cache,
   real-model tests, and the configured four-layer architecture.
8. Completed: script-level PyTorch seed, loss, validated AdamW settings, and
   warmup/cosine scheduling with boundary and state-restoration tests.
9. Next: mixed precision, then the training step with accumulation and clipping.
10. Build the pretraining loop, checkpoint/resume, validation, and reporting.
11. Configure a corpus and run a one-batch overfit test.
12. Run a small corpus smoke run; TinyStories remains a candidate.
13. Add generation and benchmark reporting, then create a release candidate.
14. Implement the SFT schema, chat template, masking, and SFT loop.
15. Run a small SFT experiment.
16. Promote stable reusable components to the decoder template.
