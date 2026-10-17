# T16 template

Self-contained 30M Tide + A16 hybrid, promoted from
`model/001-spoony-t16-50m` at `5200a64`. Despite that branch name,
the selected recipe is the existing 30M architecture.

Width 512, nine layers, SwiGLU hidden width 1152, tied embedding/head.
Each group contains two Tide layers followed by one causal attention layer.
Attention uses one query/KV head of width 16, RoPE base 10000, no sliding
window and zero dropout. Each layer is RMSNorm -> residual mixer ->
residual SwiGLU -> RMSNorm. Actual vocabulary size comes from the tokenizer.

Tide retains signed selective writes, global RMS normalization and continuous
retention/write initialization. The cleaned runtime preserves parameter paths
and seeded initialization; recurrent cache IDs use TideMixerImpl instead of
the experimental NoKVariantImpl names. Recreate old external caches.
Full training manifests from experiment branches are not resume-compatible
with this template's changed source/configuration. Load model weights or start
fresh. No experiment results or checkpoints are included.

The template retains its editable data/training/SFT defaults: pinned TinyStories
subset, vocabulary target 8192, context 256, BF16, seed 42 and 9000 updates.
At vocabulary 8192 the model has 26,527,744 parameters (30,722,048 at
vocabulary 16384). These are starting settings, not a reproduction of the source experiment's
mixed-corpus and QA pipeline. Outputs go under results/pretraining/t16.

Run from models/001-spoony with a model-local .venv:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src configs scripts tests
.venv/bin/ruff format --check src configs scripts tests
.venv/bin/python -m scripts.materialize_corpus
.venv/bin/python -m scripts.prepare_data
.venv/bin/python -m scripts.benchmark
.venv/bin/python -m scripts.run_baseline
```

Prepare fresh artifacts and select fresh output paths for each run. SFT requires
configured JSONL data and a pretrained checkpoint. See MODEL_ASSEMBLY.md,
CHECKPOINTS.md and SFT.md for contracts.
