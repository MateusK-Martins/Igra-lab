# A16 + full-width 10-token gMLP + AC-SwiGLU

Start from `model/001-spoony-a16-ac-swiglu-14m`, the conditioned model
(loss2.111418, perplexity8.259945). Its matched control scored2.111566;
the conditioning improvement was negligible in that single-seed experiment.
Train fresh, preserving the same tokenizer, seed42, six layers, width384,
SwiGLU1024, context256, BF16, AdamW schedule and9000updates.

For each layer:

1. z = A16(RMSNorm(x)); Q/K useRoPE, KV remain16D.
2. r = x + W_O z.
3. s = r + gMLP(RMSNorm(r)).
4. h = SiLU(W_G RMSNorm(s)) * W_U RMSNorm(s).
5. y = s + W_D(h * (1 + .1 tanh(W_C z))).
6. Preserve the existing final RMSNorm.

The gMLP receives384D token states, NOT compressed16D attention retrievals.
It uses the previously implemented fixed-window mechanism:384→1024 GELU,
split into two512D halves, LayerNorm on the spatial half, learned causal
10×10 token matrix plus10biases shared over channels, element-wise gating,
and512→384 decoding. No additional attention. One gMLP per layer.
W_C starts atzero and reads raw z; existing weights initialize identically
to the original AC-SwiGLU run. New gMLP initialization uses fork_rng.

Window semantics: non-overlapping10-token windows with a lower-triangular
matrix. This is the original fixed-position gMLP from our earlier experiments,
not a sliding convolution. The last partial window is zero-padded. At inference,
absolute stream count determines window position, including across cached
chunks. Training samples start at window position0 as in the earlier pipeline.

Actual parameters14,032,788 versus10,476,288 forAC-SwiGLU (+3,556,500).
This therefore compares a larger model with an added mechanism; it is not a
parameter-matched ablation. The FFN width remains1024.
A16 KV storage is unchanged32scalars/token/layer. gMLP adds fixed history;
we reuse the boundedKV cache with duplicated normalized512D spatial features:
2×10×512=10,240 scalars/sample/layer, independent of sequence length.
No new cache serialization format or training environment is introduced.

Tests retain the oldACbehavior with gMLP disabled, independently check the
token-mixing equations and all gradients, full residual/order equations,
future isolation, no cross-window mixing, snapshots, single-token/chunk
inference, constant history allocation, BF16forward/backward and parameter
counts. OldACstate dictionaries do not load strictly into this larger model.

Source and data fingerprints were frozen before training.

## Completed comparison

| Model | Parameters | Loss | Perplexity | Time (min) | Tokens/s | Peak CUDA GiB |
|---|---:|---:|---:|---:|---:|---:|
| A16 + AC-SwiGLU | 10,476,288 | 2.111418 | 8.259945 | 4.46 | 68,885 | 0.716 |
| A16 + full-width gMLP10 + AC-SwiGLU | 14,032,788 | 2.056418 | 7.817912 | 6.05 | 50,760 | 0.939 |

Loss change versus AC-SwiGLU: -0.055000. Extra parameters, nonlinear depth and normalization also change, so this cannot establish that token mixing alone caused any improvement. One seed on TinyStories, without QA/retrieval claims.

586 tests and Ruff pass. Exact seven-artifact fingerprints, training settings and unchanged initial parameters match the reference. Both process 18,429,440 target tokens. Trained full/cached inference agrees over 273 tokens in FP32/FP64. Sampled BF16 gradients and activations are finite. Full validation curves, fixed-prompt samples, gamma/gMLP activation statistics and final spatial-weight norms are retained underresults. Runtime includes validation/saving; VRAM is peak allocated memory.

### Learned gMLP and conditioning activity

- body.0: gMLP branch RMS 0.402554, attention branch RMS 0.051376, causal spatial-weight norm 0.591785, off-diagonal norm 0.581924, gamma mean absolute deviation 0.000625.
- body.2: gMLP branch RMS 0.337897, attention branch RMS 0.627527, causal spatial-weight norm 1.467620, off-diagonal norm 1.452203, gamma mean absolute deviation 0.009550.
- body.4: gMLP branch RMS 0.332070, attention branch RMS 0.483910, causal spatial-weight norm 1.407342, off-diagonal norm 1.388291, gamma mean absolute deviation 0.007904.
- body.6: gMLP branch RMS 0.299992, attention branch RMS 0.468061, causal spatial-weight norm 1.526876, off-diagonal norm 1.510157, gamma mean absolute deviation 0.006536.
- body.8: gMLP branch RMS 0.287881, attention branch RMS 0.396003, causal spatial-weight norm 1.286744, off-diagonal norm 1.244175, gamma mean absolute deviation 0.006875.
- body.10: gMLP branch RMS 0.375552, attention branch RMS 0.340757, causal spatial-weight norm 1.292019, off-diagonal norm 1.242446, gamma mean absolute deviation 0.006407.

Nonzero off-diagonal weights demonstrate learned cross-token mixing, not semantic specialization or proof that the mixing improves loss independently of other changes.

### Fixed-prompt generation

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big tree with lots of leaves. She wanted to climb the tree, but she was scared.
Lily's mom saw her and said, "Don't worry, Lily. I will help you." She took a deep breath and looked up at
