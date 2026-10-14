# Lantern 30M versus Flux V-only 30M: stories plus QA

Two independent branches: model/001-spoony-lantern-30m and
model/001-spoony-fluxv-30m. Parent models remain unchanged. Both models use
width 512, nine layers, tied 8192-vocabulary embedding/head, embedding std 0.02,
context 512, BF16, batch 8, accumulation 1, AdamW LR 3e-4 / betas (0.9,0.95) /
epsilon 1e-8 / decay 0.01, clipping 1.0, seed 42 and one shared tokenizer.

Lantern: 8 query / 2 KV heads, head width 64, RoPE base 10000, attention window
512, SwiGLU hidden width 1408; 29,565,952 parameters.
Flux V-only: tanh K, sigmoid Q/V, trainable internal RMSNorm, token-dependent
V retention sigmoid(W_D x+b_D), no K decay, SwiGLU hidden width 1152;
29,579,776 parameters. The feed-forward widths differ to match total parameter
budgets. Gate weights start at zero and bias at logit(0.99). Total sizes differ
by 13,824 parameters (0.047%), not by a large unused parameter padding. Flux
uses a parallel associative affine scan with logarithmic time depth.

## Shared data and supervision

TinyStories revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64, first 200,000
unique nonempty training stories and 1,000 held-out validation stories. Exact
stripped-text duplicates are excluded within/across splits. Prepared story
train tokens: 44,256,432; validation: 191,774. License: CDLA-Sharing-1.0.
Source: https://huggingface.co/datasets/roneneldan/TinyStories

SQuAD 1.1 revision 7b6d24c440a36b6815f21b70d25016731768db1f. Independently
seed-42 shuffle each official split, retain 30,000 eligible training questions
and 1,000 validation questions. Full passages and questions are retained; there
is no answer-centered context crop. Reject duplicate question/passage pairs,
validation contexts overlapping official train contexts, and sequences whose
prompt + first answer + end token cannot fit the shifted 512-token window.
This is a filtered subset, not a full official leaderboard evaluation. License:
CC BY-SA 4.0. Source: https://huggingface.co/datasets/rajpurkar/squad

BPE is trained only on the selected training stories and official SQuAD train
texts; no validation content is used. Both models share byte-identical prepared
artifacts. This new tokenizer differs from earlier small-model experiments, so
perplexities should be compared within this pair, not directly with old runs.

The training stream has three whole story batches followed by one whole QA
batch. Dataset indices are pre-shuffled with a fixed independent seed; the
DataLoader itself is sequential, enabling deterministic checkpoint replay.
Story loss covers all next-token targets; QA loss ignores the prompt and padding
and supervises only the first answer plus end token. Keeping whole batches
prevents short QA answers from being diluted inside a mostly-story batch.
Windows/questions are reset independently; no history crosses batch examples.

## Budget and evaluation

User budget: about 30 minutes of training per model. Timing benchmarks are
discarded; all training starts fresh. A common 6,500-update schedule is chosen
from the slower measured p95 update time with a 180-second reserve and 5%
margin, rounded down to 500 updates. Warmup 200, cosine floor 0.1. Validation
story loss every 1,000 updates; checkpoints at 2,000 / 4,000 / 6,000 / 6,500.

Each run consumes 4,875 story batches (39,000 nonrepeated selected story windows)
and 1,625 QA batches (13,000 selected questions), all batch 8. This is 19,968,000
story target tokens plus the actual supervised QA answer/end tokens. Total
processed/padded input positions are 26,624,000; those must not be called all
supervised tokens. Available corpus size and consumed supervision are distinct.

Primary results use the final-budget checkpoint, not whichever checkpoint looks
best for each task. Separately report story validation loss/perplexity on all
1,000 stories and response-only QA loss/perplexity on all 1,000 eligible QA rows.
QA loss uses the first reference plus end token. Greedy QA generation evaluates
the first 256 rows of the seeded validation subset, max 64 new tokens and no
more than 512 input tokens, with a fresh cache for each question. Report standard
SQuAD-v1-style normalized exact match and word-token F1, maximizing over all
reference aliases. No gold answer is fed during generation. Right-padding is
ignored in loss and cannot affect earlier real tokens in these causal models.

Report model sizes, actual supervised-token counts, wall time, VRAM, both sets'
metrics and QA predictions. Evaluation time is recorded separately from training.
This is one seed and a short training budget, not a convergence or statistical
architecture ranking. Raw/prepared data and weights stay ignored and local.

## Commands

From each model directory:

```bash
.venv/bin/python -m scripts.materialize_corpus
.venv/bin/python -m scripts.prepare_comparison
.venv/bin/python -m scripts.benchmark_comparison
.venv/bin/python -m scripts.train_comparison
.venv/bin/python -m pytest -q
.venv/bin/ruff check src configs scripts tests
.venv/bin/ruff format --check src configs scripts tests
```

Prepare data once, deliberately copy the fixed artifacts to the other branch,
and verify all hashes. Source is copied independently into each branch, without
a live shared-code dependency. Resume uses the existing checkpoint API and
recorded recipe commit; no partial pretraining checkpoint initializes this pair.
