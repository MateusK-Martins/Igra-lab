# Controlled RoPE reruns of compressed attention

Redo A16, A4, SMA4, independent-key SMA4 and GELU16 with RoPE base 10000.
Original no-position branches and artifacts remain preserved. Each rerun is fresh
and independent: same backbone, parameters, initialization, data/tokenizer bytes,
seed 42, order, BF16, batch 8, context/stride 256, AdamW settings, 9000 updates /
18,429,440 target tokens, validation every 500, checkpoints every 1000.
No tuning after a completed result. No Tide changes or training in this phase.

A16/A4 rotate Q and K normally; V is unchanged.
Let R_t denote the existing adjacent-pair rotary matrix at absolute position t.
For all matrix variants: q'_t = R_t q_t and M'_i = M_i R_i^T.
Then response = M'_i q'_t = M_i R_i^T R_t q_t, depending on relative displacement.
Only matrix columns rotate, not output/value coordinates. Independent keys also
use k'_i = R_i k_i; scores = (q'_t)^T k'_i / 2. Original SMA4 scores u^T response/2.
GELU16 still transforms each response before weighted aggregation. A common
position shift cancels in scores and responses. This is an explicitly chosen
matrix extension of RoPE, not a claim that it is the unique possible extension.

Inference caches store already-rotated keys/matrices, and absolute positions
start at each cache's pre-write count. Cached history is never rotated again.
Cache scalar counts and all trainable parameter counts are unchanged. Cache IDs
include '-rope-' to distinguish unrotated snapshots. Training manifest identity
also distinguishes old/no-RoPE checkpoints; these runs do not warm-start.

The optional rope_base definition setting defaults to None so existing modes
retain exact behavior. It consumes no RNG: for the same mode all initial
parameter values are bitwise equal with/without RoPE. The shared backbone also
retains its common initialization across all five mechanisms.

560 tests pass before training, including independent scalar-angle rotation
references for scores/matrix readouts and all gradients, FP64 gradcheck, common
position-shift invariance, cached offsets/snapshot restoration, causal future
isolation, BF16 CUDA backward/CPU-cache reads, unchanged initial weights/counts,
and existing regression tests. Ruff lint/format checks pass.

Home has insufficient free space for all new checkpoints. The run's ignored
results/.../checkpoints path links to /var/tmp/igra-rope-14m/<mode>, on the root
partition with adequate disk space. No old checkpoints are removed. Validation
and checkpoint intervals are unchanged. /var/tmp may be subject to system cleanup;
these local large artifacts should be moved before any temporary-file cleanup.
Only checkpoint storage location changes, not the training protocol. Metrics,
manifests, probes and fixed-prompt samples are retained inside the model branch.

Compare each rerun to its own no-RoPE counterpart first. Lantern is a historical
reference with more heads/parameters. The original runs did learn; this rerun
measures whether explicit relative positions help under the established budget.
Single-seed TinyStories results cannot establish reasoning/retrieval capability.

## Completed controlled results

Each run completed 9000 fresh updates / 18,429,440 target tokens. Initial/final gradients and logits are finite. Dataset/tokenizer fingerprints and initial shared-backbone weights match each no-RoPE counterpart; source hashes match the frozen recipe.

| Model | Parameters | Loss without RoPE | Loss with RoPE | Difference | PPL with RoPE | Minutes | Tokens/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a16 | 10,375,680 | 2.271129 | 2.170240 | -0.100888 | 8.760389 | 3.59 | 85,589 | 0.560 |
| a4 | 10,265,088 | 2.488885 | 2.379617 | -0.109268 | 10.800760 | 3.74 | 82,115 | 0.571 |
| sma4 | 10,285,968 | 2.677201 | 2.471999 | -0.205202 | 11.846106 | 4.35 | 70,630 | 0.611 |
| sma4-key | 10,295,160 | 2.508731 | 2.398993 | -0.109738 | 11.012086 | 4.43 | 69,268 | 0.620 |
| sma4-gelu16 | 10,323,288 | 2.503748 | 2.375902 | -0.127847 | 10.760712 | 6.39 | 48,083 | 0.996 |

Negative loss differences favor RoPE. This paired comparison measures adding rotary positions to each specified mechanism; it does not establish that the unrotated models could not learn. Absolute-position shift cancels in the tested matrix extension. Independent-key matrix variants have positional changes in both selection and responses.

Historical Lantern loss 2.018320 / perplexity 7.525675 is contextual: it has GQA and 12,440,064 parameters. The current models use single-head compressed retrieval. Ranking across different mechanisms does not isolate only positional encoding, and these single-seed TinyStories runs do not establish reasoning/retrieval capability.

All 560 tests and Ruff checks passed before each run. Mathematical tests independently construct rotary matrices; references cover outputs, all parameter gradients, relative shifts, causal isolation and cache offsets/restoration. Final trained FP32/FP64 full-versus-cached decoding also passed; see trained-cache-equivalence.json.

Timing includes training/validation/checkpoint saves and excludes discarded benchmark updates, probes and generations. Checkpoints were saved on the root partition through ignored links; cross-run storage conditions and laptop clocks limit strict speed interpretation. Curves and all fixed generations are in training-report.json. Samples follow the historical greedy/best-saved-checkpoint protocol; table metrics use update 9000.

## This branch: sma4

Fixed-batch gradient norm before clipping: 1.957566 initially → 5.647428 finally. Clipping remains 1.0. Readout RMS, query RMS, attention entropy and matrix statistics are retained in the probes. Trained cache check passed: FP32 maximum logit difference 1.48e-05; FP64 4.09e-14.

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, shiny rock in the water. She picked it up and started to climb.
Lily's mom said, "Lily, it's time to go home." Lily was sad because she didn't want to share her toys. She went to
