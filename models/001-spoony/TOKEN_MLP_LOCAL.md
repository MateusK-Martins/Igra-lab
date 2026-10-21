# TokenMLP with block-local embeddings

The bottleneck is 64. Each of the three TokenMLP branches owns its own pair
of token embedding tables, registered as `body.<index>.branch.embeddings`.
The down/up projections remain shared by all three branches. Original Lantern
attention, normalization, input embedding/tied head and retained SwiGLU blocks
are unchanged.

For TokenMLP occurrence l and token ID t:

    a = A_l[t]                         # 64 features
    b = B_l[t]                         # 64 features
    z = W_down x                       # 384 -> 64, shared
    output = x + W_up (tanh(a b^T) z)   # 64 -> 384, shared

Assembly creates fresh tables for every expanded occurrence, even when Repeat
reuses the same definition object. Each table uses that definition's init_std
(0.02 in the configured model). Lookups happen within each runtime branch;
tables and gradients are independent across branches. Projection gradients
still accumulate from all three branches. CUDA fused execution and activation
checkpointing are retained.

At vocabulary 8192, unique parameters are:

| Component | Parameters |
| --- | ---: |
| Input embedding / tied head | 3,145,728 |
| Six block-local token tables | 3,145,728 |
| Shared down/up projections | 49,152 |
| Six original GQA branches | 2,211,840 |
| Three original SwiGLU branches | 3,538,944 |
| Twelve RMSNorms | 4,608 |
| Total | 12,096,000 |

Source/config: `src/model/token_mlp.py`, `src/model/definitions.py`,
`src/model/assembler.py`, `configs/model/model.py`.
Training outputs are configured to the fresh directory
`results/pretraining/lantern-token-mlp-local64-v1/` to preserve previous runs.
The completed run is recorded below. Previous shared-table checkpoints have
incompatible parameter names and shapes; their results remain documented in
TOKEN_MLP.md and TOKEN_MATRIX.md.

Tests cover independent table storage/lookup/gradients, shared projection
reference gradients, parameter count, independent per-definition initialization,
CUDA forward/backward, cached logits and checkpoint restoration.


## Completed training — 2026-10-21

All 9,000 updates completed in **354.22 seconds (5m 54s)**, including validation
and checkpoint saving, consuming 18,429,440 target tokens. Final/best saved
validation loss: **2.065811**; perplexity: **7.891693**, at step 9,000.
Peak allocated CUDA memory was 736,945,152 bytes. Nine checkpoints were saved.
Strict restoration verified finite parameters, six distinct learned (8192, 64)
tables, shared projection weights (64, 384)/(384, 64), and original retained
SwiGLU gate weights (1024, 384).

| Variant | Parameters | Validation loss | Perplexity | Training time |
| --- | ---: | ---: | ---: | ---: |
| Original Lantern | 12,440,064 | 2.018320 | 7.525675 | 431.33 s |
| Shared tables, width 64 | 9,998,848 | 2.062081 | 7.862314 | 485.80 s |
| Shared tables, width 206 | 12,434,432 | 2.053480 | 7.794982 | 421.85 s |
| Block-local tables, width 64 | 12,096,000 | 2.065811 | 7.891693 | 354.22 s |

Separate tables did not improve validation quality in this single-seed run.
All rows use the same recorded TinyStories subset and training-token count.
The shared-width-64 run preceded CUDA fusion, so its wall time cannot isolate
the cost of sharing tables. The current run retains the fused implementation.
Best checkpoint:
`results/pretraining/lantern-token-mlp-local64-v1/checkpoints/step-00009000.pt`.
Plain-text progress and the final report are mirrored into the main workspace
under `models/001-spoony/results/pretraining/lantern-token-mlp-local64-v1/`.
