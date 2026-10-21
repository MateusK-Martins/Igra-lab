# All feed-forward branches replaced by shared TokenMLP

All six layers keep original Lantern RMSNorm -> residual GQA -> residual
feed-forward -> RMSNorm order, but every feed-forward branch is now TokenMLP.
There are no SwiGLU branches. Attention, width 384, GQA dimensions, RoPE,
window 256, normalization, and the tied input embedding/LM head are unchanged.
The bottleneck is 64. Exactly two token tables and two bias-free projections
are shared by all six TokenMLP occurrences, returning to shared embeddings.

    a = A[token_id]                    # shared, 64 features
    b = B[token_id]                    # shared, 64 features
    z = W_down x                       # shared, 384 -> 64
    output = x + W_up (tanh(a b^T) z)   # shared, 64 -> 384

Table lookups happen once per model forward. Each branch receives the same
looked-up vectors and projection modules, applied to its current hidden state.
All six occurrences accumulate gradients into the same four parameters.
No token vectors or matrices persist between calls. The CUDA fused formula
and activation checkpointing are retained.

## Parameters at vocabulary 8192

| Component | Unique parameters |
| --- | ---: |
| Input embedding / tied head | 3,145,728 |
| Two shared token tables | 1,048,576 |
| Shared down/up projections | 49,152 |
| Six original GQA branches | 2,211,840 |
| Twelve RMSNorms | 4,608 |
| Total | 6,459,904 |

No other widths are increased to compensate for removing the three SwiGLUs.
Editable architecture is `configs/model/model.py`; TokenMLP definition/runtime
are `src/model/definitions.py` and `src/model/token_mlp.py`.

## Run protocol

Fresh output directory: `results/pretraining/lantern-token-mlp-all-shared64-v1/`.
The same TinyStories subset and tokenizer, seed 42, batch 8, context 256, BF16,
AdamW LR 3e-4, 300-update warmup and 9,000 updates are retained. Validation
is every 500 updates; checkpoints every 1,000. The process has a 585-second
timeout plus at most five seconds to terminate. Benchmark weights are discarded.
Previous variant outputs are preserved. Existing checkpoints require their
original model configuration; removal of SwiGLUs changes state-dict contents.

Tests cover all six replacements, lookup once per table, the six-use shared
projection/table gradient reference, parameter count, cached inference, strict
state restoration and CUDA forward/backward. Historical variants and their
results are recorded in TOKEN_MATRIX.md, TOKEN_MLP.md and TOKEN_MLP_LOCAL.md.

## Completed run — 2026-10-21

All 499 tests passed; Ruff and `git diff --check` passed. The full run completed
9,000 updates in **308.00 seconds (5m 8s)** including validation and checkpoint
saving, consuming **18,429,440 target tokens**. Final/best saved validation
loss: **2.242750**; perplexity: **9.419195**, at step 9,000. Peak allocated CUDA
memory was 605,611,008 bytes. Nine checkpoints were saved.

Strict checkpoint restoration verified finite weights, six TokenMLP runtimes,
zero SwiGLUs, exactly two shared (8192, 64) tables and shared down/up projections.
Recorded corpus identity and consumed token count match the earlier variants.
Best checkpoint:
`results/pretraining/lantern-token-mlp-all-shared64-v1/checkpoints/step-00009000.pt`.

| Model | Parameters | Final validation loss | Perplexity | Training time |
| --- | ---: | ---: | ---: | ---: |
| Original Lantern | 12,440,064 | 2.018320 | 7.525675 | 431.33 s |
| Three shared TokenMLPs, width 64 | 9,998,848 | 2.062081 | 7.862314 | 485.80 s |
| Six shared TokenMLPs, width 64 | 6,459,904 | 2.242750 | 9.419195 | 308.00 s |

Replacing every feed-forward branch reduced size and wall time, with worse
validation quality in this single-seed experiment. The first three-block run
preceded CUDA fusion, so its timing does not isolate block count alone.
Logs and the final report are mirrored into the main workspace under
`models/001-spoony/results/pretraining/lantern-token-mlp-all-shared64-v1/`.
