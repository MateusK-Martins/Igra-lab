# SMA8: separate keys and nonlinear matrix responses

Extends `model/001-spoony-sma4-gelu16-rope-14m`, preserving the six-layer
384-wide backbone, 1024-wide SwiGLU, tokenizer, seed42, context256,
BF16, optimizer and 9000-update TinyStories schedule.

Only matrix latent width changes 4→8. Every token generates an unconstrained
8×8 matrix and independent 8D key. Current query is 8D. RoPE rotates Q/K and
matrix column coordinates (M_i R_i^T). Causal scores are q·k/sqrt(8).
Each matrix response M_i q is projected 8→16 with GELU BEFORE softmax-weighted
aggregation; the final16D result is decoded to384. Bias settings unchanged.
No Tide, no extra projections beyond the inherited response projection.

Actual parameters10,453,008 versus SMA4's10,323,288 (+129,720,1.26%).
The historical '14M pipeline' names the backbone/training protocol; compressed
attention reduces its actual parameter count. FFN width is not enlarged.
Cache72 scalars/token/layer (64 matrix+8 key), versus20 for SMA4 and32 forA16.
Inference cache grows with sequence length. State dictionaries reject old4D
weights; dimension-specific cache IDs and write validation reject4D caches.

Production remains vectorized; responses have shape[B,T,S,8] and the nonlinear
projection[B,T,S,16]. This is quadratic attention, not a recurrent scan.
Tests independently check equations/all gradients, RoPE, future isolation,
FP64gradcheck, chunks/single tokens/snapshots, shape mismatch withoutmutation,
BF16forward/backward CPU/CUDA and oldSMA4compatibility.

Frozen source commit and data hashes are recorded per run.

## Completed comparison

| Model | Parameters | Loss | Perplexity | Time (min) | Tokens/s | Peak CUDA GiB |
|---|---:|---:|---:|---:|---:|---:|
| SMA4 + keys + GELU16 + RoPE | 10,323,288 | 2.375902 | 10.760712 | 6.39 | 48,083 | 0.996 |
| SMA8 + keys + GELU16 + RoPE | 10,453,008 | 2.249660 | 9.484512 | 6.67 | 46,028 | 1.048 |
| A16 + RoPE | 10,375,680 | 2.170240 | 8.760389 | 3.59 | 85,589 | 0.560 |

SMA8 minus matched SMA4 loss: -0.126242. The wider operator changes both query/key width and matrix capacity, so this does not isolate matrix size alone. Compare A16 cautiously: its fixed-value attention has different cache and parameter costs. Neither loss nor matrix singular values prove associative retrieval or reasoning capability. This is one seed on TinyStories.

All571tests and Ruff pass. Exact seven-artifact fingerprints, training settings and unchanged-backbone initialization match SMA4. Both process18,429,440 target tokens. FP32/FP64 trained full/cached inference agrees over273tokens. BF16 activation/gradient probes are finite. Full validation curves, fixed-prompt samples, initial/final per-layer RMS, entropy and singular-value diagnostics are retained inresults. CUDA memory is allocated memory, not total device use. Runtime includes validation/saving.

### Sample

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's friend, Timmy, came over to play. "Hi, Lily! What are you doing?" asked Lily. "I'm playing with my
