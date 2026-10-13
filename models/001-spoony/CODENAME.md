# Codename: attention baseline

This branch (`codex/codename`) freezes a larger Spoony baseline before an
attentionless comparison. Model-local paths remain `models/001-spoony/`.
It starts from committed Spoony `6d2d5f3` and deliberately carries over the
working-tree Embedding.init_std implementation and tests plus the 0.02
initialization choice, together with the current RMSNorm-before-attention
recipe. Other uncommitted Spoony work is not included.

## Frozen recipe

- Six layers, width 384; each layer retains Spoony's RMSNorm -> residual GQA ->
  residual SwiGLU -> RMSNorm order. No architecture redesign beyond scaling.
- Eight query heads, two KV heads, head width 48, RoPE base 10000, attention
  window 256, dropout zero. Context and stride are both 256.
- SwiGLU hidden width 1024, RMSNorm epsilon 1e-6, tied embedding/head,
  embedding normal initialization standard deviation 0.02.
- Train-only byte-level BPE, vocabulary 8192; total 12,440,064 parameters.
- TinyStories original text files, revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, first 30,000 unique nonempty
  training stories and first 256 unique validation stories. Exact stripped-text
  SHA-256 deduplication runs within and across splits, training first. This is a
  deterministic prefix subset, not a random representative sample; near-duplicate
  stories can remain. Upstream license: CDLA-Sharing-1.0.
- 6,421,279 prepared train tokens; 55,421 validation tokens. End-of-document
  tokens are appended; next-token windows can cross document boundaries.
- Batch 8, accumulation 1, BF16 on RTX 4050 Laptop 6GB; seed 42; shuffled
  training, ordered validation, num_workers=0, four PyTorch CPU threads.
- AdamW LR 3e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01;
  9,000 updates, 300-update warmup, cosine floor 0.1, gradient clipping 1.0.
- Validation every 500 updates; checkpoints every 1,000 and at completion;
  TensorBoard every 10. Best checkpoint means best among saved checkpoints.

The clean preliminary benchmark measured approximately 0.043 seconds/update;
9,000 updates target roughly 6.5 minutes of compute, with room below ten minutes
for validation and checkpoint saving. This is a measured estimate, not a hard
wall-clock termination policy. Download, BPE preparation, tests and benchmarking
are outside the training budget. The fresh run excludes benchmark-trained weights.

## Commands

Run from this model directory with its `.venv` interpreter:

```bash
.venv/bin/python -m scripts.materialize_corpus
.venv/bin/python -m scripts.prepare_data
.venv/bin/python -m scripts.benchmark
.venv/bin/python -m scripts.run_baseline
.venv/bin/python -m pytest -q
.venv/bin/ruff check src configs scripts tests
.venv/bin/ruff format --check src configs scripts tests
```

Materialization and preparation reject existing artifacts. Training saves under
`results/pretraining/codename-tinystories-v1/`; do not reuse that directory for a
fresh run. Prepared data, raw data, environments, TensorBoard and checkpoints
remain ignored. The small benchmark/report JSON files are versioned. Reproduce
on another machine with a model-local environment using requirements/local-cu128.txt.

## Attentionless comparison contract

Preserve the prepared tokenizer and split artifact hashes, the same seed and
DataLoader order, context length, loss and evaluation protocol. Replace the GQA
branch, preserve the surrounding model, and report actual parameter counts;
match total unique parameters as closely as the candidate permits. Any longer
recurrent history or different reset policy must be declared separately.

Use 9,000 updates (approximately 18.4M consumed target tokens) for equal-token
comparison. Rebenchmark the candidate separately for equal-time comparison,
including validation and saving. Report validation loss, perplexity with this
same tokenizer, throughput and peak allocated VRAM. Fixed greedy prompts are
in configs/inference_baseline.py. One seed is preliminary; repeat seeds before
making architecture claims. This short run does not establish convergence or
broad language capability.

## Completed baseline — 2026-10-13

Recipe commit: `2eb1a7badd44a58a59841a5e817effb218fc071e`. All 480 tests
passed, including configured-model forward/backward, cache equivalence across
the enlarged window, embedding initialization, corpus deduplication and
checkpoint/resume coverage. Ruff lint and formatting checks passed.

Training completed in **431.33 seconds (7 minutes 11 seconds)**, including
validation, checkpoint saving, manifest creation and writer closure, excluding
data download/preparation, benchmarking and the subsequent generation samples.
The run consumed **18,429,440 target tokens** in 9,000 updates: two full epochs
plus 2,728 batches. Partial final batches account for the difference from
9,000 × 2,048 tokens. Overall throughput was about 42,726 target tokens/second;
peak allocated CUDA memory was 802,710,528 bytes (0.748 GiB; allocator usage,
not total device occupancy).

Initial validation loss was 9.242885; final loss **2.018320**, perplexity
**7.525675**. Every recorded validation measurement improved; convergence is
not established. The final checkpoint is also the best saved checkpoint:
`results/pretraining/codename-tinystories-v1/checkpoints/step-00009000.pt`.
Nine checkpoints were saved. The final checkpoint was loaded successfully for
generation and its settings were checked against the saved manifest.

The fixed greedy samples produce readable sentences but exhibit repetition
and inconsistent story progression. The full samples, 18 validation measurements,
subset identities and actual timing are in
[training-report.json](results/pretraining/codename-tinystories-v1/training-report.json).
The [manifest](results/pretraining/codename-tinystories-v1/manifest.json) pins
processed artifacts, settings, source hashes and environment. Prepared data and
checkpoints remain local and ignored; retain this worktree for the comparison.

The next experiment can now replace attention on a separate branch while
reusing these exact data/tokenizer artifacts. Treat these as single-seed
preliminary results. Checkpoint compatibility includes the recipe Git commit;
result-only commits do not change the model weights, but rebuilding a resume
manifest on a different commit is rejected by the existing strict loader.
