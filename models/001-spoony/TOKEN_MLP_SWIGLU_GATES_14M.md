# Shared TokenMLP with per-layer SwiGLU gates — Lantern 14M test

Branch: `codex/lantern-tokenmlp-swiglu-gates-14m`.

This experiment keeps the all-shared baseline's two token tables, shared
384-to-64 down projection, tanh outer-product matrix, and shared 64-to-384
output projection. Each TokenMLP layer now owns a separate bias-free 64-to-64
gate projection. Given the matrix-transformed bottleneck vector `q`, the layer
computes `q_gated = SiLU(G_layer(z)) * q` before the shared output projection
and residual addition. The gate projection and matrix output are the two
SwiGLU-style paths; the gate is layer-local while the value path uses the
existing token-specific matrix.

At vocabulary 8,192 the six gate projections add 24,576 parameters to the
6,459,904-parameter all-shared baseline, for 6,484,480 total. Training used the
same tokenizer, TinyStories subset, seed, optimizer, 9,000 updates, and
18,429,440 target tokens as the baseline.

| Run | Parameters | Updates / tokens | Time | Best validation loss | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| All six layers TokenMLP | 6,459,904 | 9,000 / 18,429,440 | 308.0 s | 2.242750 | 9.4192 |
| Per-layer SwiGLU-style gate | 6,484,480 | 9,000 / 18,429,440 | 306.7 s | 2.311159 | 10.0861 |

In this single-seed run, the gated version finished 0.06841 validation-loss
points higher with effectively unchanged training time. The comparison does
not establish a multi-seed difference. The report includes the validation
curve, subset hashes, and greedy samples; benchmark and report are stored in
`results/pretraining/lantern-tokenmlp-swiglu-gates-14m-v2/`.
