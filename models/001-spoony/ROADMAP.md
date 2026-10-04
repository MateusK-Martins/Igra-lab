# Spoony roadmap

Spoony is a small, independently runnable decoder Transformer. Its size is
chosen for fast local training, but its data, training, evaluation, cloud, and
supervised fine-tuning systems must use scalable research-quality contracts.

TinyStories is the first pretraining corpus used to validate the system. It
does not define the project’s long-term scope.

## Project rules

- Every experiment choice belongs in an explicit frozen dataclass, constructed
  by a named preset function. It is never hidden in training code.
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
├── assets/
│   └── tokenizers/
├── data/
│   ├── manifests/
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

Configuration is Python, using frozen dataclasses. The dataclasses are the
typed contracts, and named functions build complete experiment settings:

```text
src/config/
├── types.py       # Dataclasses and their validation rules
└── snapshot.py    # Save a resolved configuration as JSON

configs/
├── data/
│   └── datasets.py    # Selected sources and manifests
└── tokenizers/
    └── preparation.py # Tokenizer settings and output directory
```

`dataclasses.replace` creates a deliberate variation of a preset without a
custom merge language. Constructors validate types and ranges immediately. A
run saves its final configuration as `resolved-config.json` in both its run
directory and checkpoints.

Local and cloud training differ through named presets, not separate training
implementations.

## Data system

Source modules:

```text
src/data/
├── schemas.py
├── sources.py
├── split.py
├── tokenizer.py
├── prepare.py
├── serialization.py
├── chunker.py
└── dataloader.py
```

Responsibilities:

- `schemas.py`: validated document and metadata structures.
- `sources.py`: download or stream a declared source.
- `split.py`: deterministic document-level train and validation splits.
- `tokenizer.py`: train, save, load, encode, and decode BPE.
- `prepare.py`: source to split to tokenizer to serialized token artifacts.
- `serialization.py`: efficient local read and write of prepared token data.
- `chunker.py`: next-token windows from a token stream.
- `dataloader.py`: batches, shuffling, workers, and pinned memory.

Dataset manifests live in `data/manifests/` and record the source, revision,
license, citation, expected checksum when available, split policy, document
field, language, sample limit, and deterministic sampling seed.

The initial TinyStories adapter should use the official training and validation
splits. For a source without a supplied validation split, split by document
before training the tokenizer.

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
assets/tokenizers/<tokenizer-id>/
├── tokenizer.json
└── metadata.json
```

Metadata records the target and actual vocabulary sizes, special-token IDs,
training-manifest hash, training-document count, library version, and Git
commit. The model always uses the actual vocabulary size learned by the
tokenizer artifact.

## Transformer model

Target modules:

```text
src/model/
├── config.py
├── embeddings.py
├── rope.py
├── normalization.py
├── attention.py
├── feedforward.py
├── transformer.py
├── cache.py
└── initialization.py
```

The architecture is a decoder-only Transformer:

```text
Token embeddings
    ↓
RoPE positional encoding
    ↓
Repeated Transformer blocks
    ├── RMSNorm
    ├── causal self-attention
    ├── residual connection
    ├── RMSNorm
    ├── SwiGLU feed-forward network
    └── residual connection
    ↓
Final RMSNorm
    ↓
Vocabulary projection
```

`ModelConfig` controls vocabulary size, context length, model width, layer
count, attention heads, key/value heads, feed-forward multiplier, dropout,
RoPE theta, embedding tying, attention implementation, and dtype.

Implement standard multi-head attention first, while keeping `n_kv_heads` in
the configuration so grouped-query attention can be enabled later. Prefer
PyTorch scaled dot-product attention with a readable fallback for research and
debugging.

## Pretraining

Target modules:

```text
src/training/
├── seed.py
├── state.py
├── loss.py
├── optimizer.py
├── scheduler.py
├── precision.py
├── distributed.py
├── checkpoint.py
├── logging.py
├── loop.py
└── train.py
```

Required capabilities:

- AdamW.
- Warmup and cosine learning-rate schedule.
- Gradient clipping and gradient accumulation.
- Configurable FP16 or BF16 mixed precision.
- Gradient scaler when FP16 is selected.
- TensorBoard metrics.
- Periodic validation loss, perplexity, and fixed-prompt generation.
- Checkpoint creation and exact resume.
- Deterministic seeds for Python, PyTorch CPU, CUDA, and DataLoader workers.
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
preparation settings live in `configs/tokenizers/`.

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

The same training code supports local and cloud runs:

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

1. Build and test typed dataclass configuration contracts and presets.
2. Add the TinyStories manifest and source adapter.
3. Implement document-aware preparation, BPE artifact generation, and token
   serialization.
4. Add DataLoader construction.
5. Implement model configuration and embeddings.
6. Implement RMSNorm, RoPE, attention, SwiGLU, and Transformer blocks.
7. Add shape and causal-mask tests.
8. Implement the pretraining loop.
9. Run a one-batch overfit test.
10. Run a TinyStories smoke run.
11. Add resume, validation, generation, and benchmark reporting.
12. Create the first release candidate.
13. Implement the SFT schema, chat template, masking, and SFT loop.
14. Run a small SFT experiment.
15. Promote stable reusable components to the decoder template.
