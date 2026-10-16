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

Results pending. Source and data fingerprints will be frozen beforetraining.
