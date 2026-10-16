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

Results pending. Frozen source commit and data hashes are recorded per run.
