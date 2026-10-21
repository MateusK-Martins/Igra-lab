# Lantern shared token matrices

Based on `model/001-spoony-lantern`. Six layers, width 384, original GQA/RoPE
and normalization order. Layers 1, 3 and 5 retain residual SwiGLU; layers 2, 4
and 6 replace it with a projected token-matrix branch.

For token ID t, two learned tables supply a_t and b_t in R^64:

    z = W_down x                    # 384 -> 64
    M_t[i,j] = tanh(a_t[i] * b_t[j]) # 64 x 64
    output = x + W_up (M_t z)       # 64 -> 384, then residual

The bias-free down/up projections are shared across all three replacement
layers, as are both token embedding tables. They are independent of the input
embedding and untied to each other. No extra activation is inserted around the
projections; the nonlinear matrix operation uses tanh after the outer product.
The matrix has 36 times fewer elements than the initial 384 x 384 version.
The same token ID selects the same matrix across positions and layers; the
hidden state changes with context and depth. Both tables use normal standard
deviation 0.02. Projections retain PyTorch's default Linear initialization.

Assembly registers exactly two tables and two projections on the model.
Table lookup occurs once per forward. Each matrix block applies the same down
and up projection to its own hidden state. Gradients accumulate from every use.
No IDs, vectors, or matrices persist between calls; cached inference uses the
current input IDs. Chunking/checkpointing avoids retaining dense matrices for
backward; chunk size 2048 processes the configured batch in one chunk.

Definitions and runtime: `src/model/definitions.py`, `src/model/token_matrix.py`.
Editable dimensions and recipe: `configs/model/model.py`.

## Parameter budget at vocabulary 8192

| Component | Unique parameters |
| --- | ---: |
| Input embedding / tied head | 3,145,728 |
| Two shared 64-dimensional matrix embedding tables | 1,048,576 |
| Shared 384 -> 64 and 64 -> 384 projections | 49,152 |
| Six GQA branches | 2,211,840 |
| Three SwiGLU branches (original hidden width 1024) | 3,538,944 |
| Twelve RMSNorms | 4,608 |
| Total | 9,998,848 |

Actual total is 5,804,544 + 512 * vocabulary size. Attention and retained
SwiGLU blocks use the original Lantern dimensions, including SwiGLU hidden
width 1024. Only the replaced feed-forward branches change; the lower total
parameter count is intentional. The projected variant is not checkpoint-compatible with
the earlier full-width variant or original Lantern.

## Runs and verification

Outputs for this projected architecture use
`results/pretraining/lantern-token-matrix-14m-v3/`. The TinyStories tokenizer,
corpus, batch 8 and context 256 remain inherited from Lantern. Earlier v1/v2
runs used full-width matrices and were interrupted; their measurements do not
describe this architecture. Historical CODENAME.md metrics belong to Lantern.

All 492 tests pass, including explicit matrix output/gradient reference,
projected output and gradient accumulation through repeated shared tables and
projections, parameter budget, cached-vs-full logits, save/load restoration,
configuration validation and CUDA mixed precision. Run tests from this model
directory with `.venv/bin/python -m pytest -q`.

Training prints plain-text progress every 10 updates and validation metrics at
each configured interval. The training launch has a process timeout under ten
minutes. The update schedule must be selected from this architecture's own
benchmark; shortening the experiment and speeding up a step are distinct changes.

## Final architecture benchmark — 2026-10-21

With original Lantern attention and retained SwiGLU width 1024, 100 measured
updates averaged 0.05076 seconds each (p95 0.05172), following 10 warmup updates.
Validation took 0.642 seconds; peak allocated CUDA memory was 708,109,312 bytes.
Actual parameter count: 9,998,848. The full 9,000-update schedule was retained,
with 300 warmup updates and a process timeout of 585 seconds plus at most five
seconds for forced termination. Benchmark weights were discarded before the run.
Plain-text logs are mirrored to the main workspace at
`models/001-spoony/results/pretraining/lantern-token-matrix-projected/training-progress.txt`.

## Completed projected run — 2026-10-21

The final architecture completed all 9,000 updates in **485.80 seconds (8m 6s)**,
including validation and checkpoint saving. It consumed 18,429,440 target tokens.
Final validation loss: **2.062081**; perplexity: **7.862314**. Best saved checkpoint:
`results/pretraining/lantern-token-matrix-14m-v3/checkpoints/step-00009000.pt`.
Nine checkpoints were saved. Peak allocated CUDA memory: 710,206,464 bytes.
Final checkpoint verification confirmed finite weights, two shared (8192, 64)
tables, projection weights (64, 384)/(384, 64), and retained SwiGLU gate weights
(1024, 384). No widening of retained feed-forward blocks was used.
This is one TinyStories prefix-subset experiment, without a quality advantage
claim. Some greedy samples repeat phrases; samples and the full validation
curve are in `training-report.json`.
