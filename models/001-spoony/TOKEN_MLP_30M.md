# Lantern 30M attention with full shared TokenMLP

Branch: `codex/lantern-30m-token-mlp-shared64`, based on
`model/001-spoony-lantern-30m`. The smaller TokenMLP branch/worktree is preserved.

All nine SwiGLU branches are replaced by residual TokenMLP. Original Lantern
30M dimensions are retained: width 512, nine layers, eight query heads, two KV
heads, head width 64, RoPE base 10000, window/context 512, RMSNorm epsilon 1e-6,
input embedding standard deviation 0.02, and tied LM head. Two learned token
tables of width 64 and two bias-free 512 -> 64 -> 512 projections are shared
across all nine TokenMLP occurrences. Token table initialization is N(0, 0.02).

    a, b = A[token_id], B[token_id]
    output = x + W_up (tanh(a b^T) W_down x)

No widths are increased to compensate for removing FFNs. At vocabulary 8192:

| Component | Unique parameters |
| --- | ---: |
| Input embedding / tied head | 4,194,304 |
| Two shared token tables | 1,048,576 |
| Shared down/up projections | 65,536 |
| Nine original GQA branches | 5,898,240 |
| Eighteen RMSNorms | 9,216 |
| Total | 11,215,872 |

Original Lantern 30M has 29,565,952 parameters. Runtime uses the same fused CUDA
TokenMLP formula as the smaller experiment. Attention uses built-in causal SDPA
for uncached chunks covered by the window, mathematically equivalent to the
former explicit mask. Cached calls and chunks extending beyond the window keep
the explicit mask and write-before-read cache semantics. Attention architecture
and dimensions are unchanged. Backend rounding may differ. Speed differences
from the original baseline include this execution optimization.

## Exact restored data and baseline protocol

The original prepared files were no longer local. Restored the archived
`prepare_comparison.py` workflow and its pinned SQuAD preparation constants.
Downloaded/prepared 200,000 training stories, 1,000 held-out stories, 30,000
training QA examples and 1,000 held-out QA examples. Every artifact SHA-256
matches the original Lantern 30M manifest: tokenizer, story tokens/documents/
metadata, and both QA files. Sources and licenses are recorded in
COMPARISON_30M.md and the original preparation report.

Train on the same seeded ConcatDataset with batch 8, context/stride 512, BF16,
seed 42, AdamW LR 3e-4 and the original 6,500-update schedule with 200 warmup
updates. Validation every 1,000; checkpoint every 2,000 and at completion.
26,624,000 padded input positions are processed. Masked QA labels mean this is
not the count of supervised targets; the original exposure report records
19,850,424 supervised targets for this same shuffled sequence.

Fresh outputs: `results/pretraining/lantern-30m-token-mlp-shared64-v1/`.
Main-workspace mirrors: `models/001-spoony/results/pretraining/lantern-30m-token-mlp-shared64-v1/`.
Training has a 585-second process timeout plus at most five seconds to terminate.
Preparation, benchmark and separate final QA loss evaluation are outside the
training timing. Benchmark weights are discarded before the fresh run.

## Validation before training

503 tests pass, including configured nine-layer gradients, cache equivalence
across the 512-token boundary, shared tables/projection gradient accumulation,
checkpoint restoration, causal fast-path output/gradient references and CUDA
mixed precision. Ruff and `git diff --check` pass. The initial benchmark was
0.15224 seconds/update; the equivalent causal SDPA path reduced the measured
100-update mean to 0.07068 seconds/update (p95 0.07032), with validation taking
1.80 seconds and peak allocated CUDA memory 1,179,344,896 bytes. This benchmark
includes the actual mixed story/QA loader.

## Completed run — 2026-10-21

Completed all **6,500 updates in 473.84 seconds (7m 54s)**, including loading,
validation and checkpoint saving. Consumed **26,624,000 padded input positions**,
matching the original baseline. Final/best saved story validation loss:
**2.284905**, perplexity **9.824748**, at step 6,500. Peak allocated CUDA memory:
1,179,344,896 bytes. Checkpoint:
`results/pretraining/lantern-30m-token-mlp-shared64-v1/checkpoints/step-00006500.pt`.

Strict restoration verified finite weights, nine TokenMLPs, no SwiGLUs, shared
(8192,64) tables, projection weights (64,512)/(512,64), and original attention
projection dimensions. New and baseline manifests have exactly matching
artifact hashes, chunking, optimizer/scheduler, loader, precision and QA-format
settings. All 503 tests passed before training; lint/format and diff checks pass.

Separate answer-only QA evaluation used all original 1,000 held-out QA examples:
loss **6.261028**, perplexity **523.757186**. QA generation EM/F1 was not evaluated.

| Model | Parameters | Story loss / PPL | QA answer loss / PPL | Training time |
| --- | ---: | --- | --- | ---: |
| Original Lantern 30M | 29,565,952 | 2.055921 / 7.814034 | 6.080587 / 437.285769 | 1228.95 s |
| Full shared TokenMLP 64 | 11,215,872 | 2.284905 / 9.824748 | 6.261028 / 523.757186 | 473.84 s |

This run trades validation quality for fewer parameters and lower memory use.
Timing also benefits from the equivalent causal SDPA execution path, so speed
cannot be attributed solely to replacing FFNs. This is a single-seed comparison.
The previous smaller experiments use a different tokenizer/data recipe and their
perplexities should not be directly compared with this run.
