# TokenMLP: parameter-matched Lantern experiment

`TokenMLP` is the name of the block introduced in TOKEN_MATRIX.md. This variant
changes only its bottleneck width, from 64 to 206; original Lantern attention
and retained SwiGLU settings are preserved. Layers 2, 4 and 6 use TokenMLP;
layers 1, 3 and 5 keep residual SwiGLU with hidden width 1024.

    z = W_down x                       # 384 -> 206
    M_t[i,j] = tanh(a_t[i] * b_t[j])    # 206 x 206
    output = x + W_up (M_t z)          # 206 -> 384, then residual

Two vocabulary tables and the two bias-free projections are each registered
once on the model and shared across all three blocks. Table initialization
standard deviation remains 0.02. Linear projections retain their default
initialization. Input embedding/head tying, normalization, GQA and RoPE match
Lantern. There are no additional projection activations.

Definitions: `src/model/definitions.py::TokenMLP`.
Runtime: `src/model/token_mlp.py::TokenMLPImpl`.
Editable settings: `configs/model/model.py` (`bottleneck_features=206`).

## Parameter matching

With vocabulary 8192, the total is 8,901,120 + 17,152 * bottleneck width.
Width 206 is the closest integer match to original Lantern's 12,440,064:

| Component | Unique parameters |
| --- | ---: |
| Input embedding / tied head | 3,145,728 |
| Two shared token tables | 3,375,104 |
| Shared down/up projections | 158,208 |
| Six original GQA branches | 2,211,840 |
| Three original SwiGLU branches | 3,538,944 |
| Twelve RMSNorms | 4,608 |
| Total | 12,434,432 |

This is 5,632 parameters (0.0453%) fewer than Lantern. Original attention and
retained feed-forward widths are not adjusted to fill the difference.

## Execution

Materializing the larger matrix was too slow for the ten-minute run limit.
CUDA now compiles a fused outer-product/tanh/multiply/reduction expression
with `torch.compile(fullgraph=True, dynamic=True)`. It evaluates the same
formula without materializing the full dense matrix in GPU memory; it does
not approximate tanh or replace the nonlinear matrix with a rank-one map.
The CPU path retains explicit matrix multiplication. Activation checkpointing
remains enabled during gradient computation. CUDA fused reduction accumulates
in FP32 for FP32 table inputs; rounding can differ from the former autocast
matrix multiplication.

The full suite includes explicit output/gradient references for the fused CUDA
operation in FP32 and BF16, shared projection/table gradient accumulation,
parameter matching, cached inference, state restoration and configuration
validation. Run from this model directory with `.venv/bin/python -m pytest -q`.

## Training

Fresh run outputs: `results/pretraining/lantern-token-mlp-matched-v1/`.
Use the same prepared TinyStories subset/tokenizer, seed 42, batch 8, context
256, AdamW 3e-4, 300 warmup updates and 9,000 total updates. Validation every
500; checkpoints every 1,000. A launch timeout limits the process to 585 seconds
plus at most five seconds to terminate. Benchmark weights are discarded.
The larger bottleneck and renamed state keys require a fresh run; previous
TokenMatrix checkpoints are preserved under their original output directories.

## Completed run — 2026-10-21

All 498 tests passed; Ruff checks and `git diff --check` passed. The run completed
all 9,000 updates in **421.85 seconds (7m 2s)** including validation and saving,
consuming **18,429,440 target tokens**. Peak allocated CUDA memory was
744,452,096 bytes. Final/best saved validation loss: **2.053480**;
perplexity: **7.794982**, at step 9,000. All nine scheduled checkpoints exist.
Strict checkpoint restoration verified finite parameters, shared tables of
shape (8192, 206), projections (206, 384)/(384, 206), and retained original
SwiGLU gate weights (1024, 384).

| Variant | Parameters | Final validation loss | Perplexity | Training time |
| --- | ---: | ---: | ---: | ---: |
| Original Lantern | 12,440,064 | 2.018320 | 7.525675 | 431.33 s |
| Previous TokenMatrix, bottleneck 64 | 9,998,848 | 2.062081 | 7.862314 | 485.80 s |
| TokenMLP, bottleneck 206 | 12,434,432 | 2.053480 | 7.794982 | 421.85 s |

The wider block improves slightly over the previous smaller variant, but the
original Lantern retains lower validation loss. CUDA fusion also changed for
this run, so the speed difference is not attributable to width alone. All runs
use matching recorded TinyStories subsets and 9,000 updates; this is one seed.
The final checkpoint is
`results/pretraining/lantern-token-mlp-matched-v1/checkpoints/step-00009000.pt`.
Plain-text progress and the final report are mirrored into the main workspace
under `models/001-spoony/results/pretraining/lantern-token-mlp-matched-v1/`.
