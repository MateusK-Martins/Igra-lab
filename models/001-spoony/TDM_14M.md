# Temporal Difference Mixer attention hybrid

Four layers at width384 with SwiGLU1024 and tied8192 embedding/head (initstd.02).
Each layer: RMSNorm -> residual Lantern GQA -> RMSNorm -> residual TDM ->
RMSNorm -> residual SwiGLU. A final RMSNorm precedes the LM head.
Attention:8query heads,2KV heads,48features/head, window256, RoPE10000, dropout0.
Exactly13,481,856parameters. No Tide mixer or learned forget gate is used.

TDM scales are(1,2,4,8,16,32). For each token/scale, D=(P_end-2P_(end-m)+P_(end-2m))/m.
Negative endpoints clamp to zero: missing past is zero-padded, and the divisor
remains m (no partial-window renormalization). Prefix sums are computed once per
call in FP32 for low-precision inputs; FP64 is preserved. Independent bias-free
full-width W_m project each difference. alpha=softmax(W_r*x) uses one bias-free
router initialized to zero (uniform scales). W_m and output W_o use default-style
Kaiming uniform initialization, no additional trainable gains. The output branch is
W_o*sum(alpha_m*W_m*D_m); the declared residual adds x once. Scale projections use
batched einsum; no Python loop over tokens. Fixed six scales give linear temporal
filter work/storage; projections costO(B*T*S*D^2). Attention retains its own cost.

Streaming TDM stores63past normalized inputs, not persistent cumulative sums, so
current+history covers64tokens. CacheStorage adds typed HistoryEntry with snapshot
format2; existing KV/recurrent format1 remains supported. History stays bounded,
but the complete hybrid attention KV cache retains the Lantern history contract.
At FP32, TDM history is94.5KiB/sample/layer (378KiB across four layers), versus
Tide's1.5KiB/sample/layer. Cached execution requires no_grad; training is functional.

Tests independently sum explicit windows, compare projected outputs and all gradients,
check constant/step/ramp responses and startup, FP64gradcheck, future isolation,
long sequences, BF16, reset, bounded history, snapshot restoration and CPU/CUDA
transfers.751tests/Ruff passed before training (before adding only the diagnostic
script). No scan optimization or training infrastructure rewrite is part of this run.

Same original14M TinyStories tokenizer/data hashes, seed42, context256,batch8,BF16,
9000updates,AdamW3e-4/betas(.9,.95)/eps1e-8/decay.01,clip1,warmup300/cosinefloor.1,
validation every500/checkpoints1000. Fresh weights; benchmark weights discarded.
Existing benchmark/run_baseline provide timing, VRAM, full validation curve and
fixed greedy samples; probe_tdm reports scale routing, norms and finite gradients.
Results:results/pretraining/tdm-hybrid-14m/.

This is not an isolated TDM ablation: compared with the prior hybrid it has four
attention/fourTDM/fourFFN blocks instead of twoattention/fourTide/sixFFN, additional
pre-normalizations, and different parameter counts. The user's selected four-layer
attention+TDM architecture keeps the experiment near14M without altering FFN width.
A matched attention-only or convolution control would be needed to attribute gains
to temporal differences specifically. No such extra run is included. Single seed,
no novelty claim, no inference of semantic state tracking from these metrics.
