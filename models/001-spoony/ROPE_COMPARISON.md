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
