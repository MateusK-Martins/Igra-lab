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

## Completed result

All9000updates/18,429,440target tokens completed from scratch. Frozen source hashes, dataset/tokenizer hashes and resolved training settings were verified after completion. No architecture/hyperparameter changes occurred during training.

| Model | Parameters | Validation loss | PPL | Training min |
|---|---:|---:|---:|---:|
| Lantern | 12,440,064 | 2.018320 | 7.525675 | 7.19 |
| Tide | 13,774,080 | 2.010153 | 7.464462 | 6.32 |
| Tide/Lantern | 13,329,408 | 1.983780 | 7.270169 | 5.02 |
| TDM/attention | 13,481,856 | 1.996956 | 7.366597 | 8.22 |

TDM peak CUDA allocated841,324,032bytes (0.784GiB); throughput37,360target tokens/s including validation/saving. Benchmark52.957ms/update versus31.953ms for the previous hybrid. Timing is a historical run comparison, not an interleaved hardware trial. The best saved checkpoint is step9000.

### Interpretation

The implemented model is numerically stable and learns language under this recipe. It improves loss slightly over pure Tide/Lantern references but is worse than the earlier Tide/Lantern hybrid and slower. This is not evidence that temporal differences themselves help or hurt: depth, attention frequency, FFN count and normalization layout also changed. A matched attention-only and causal-convolution control would be needed to isolate the mechanism; no extra run has been started. This test does not establish semantic change tracking, long-range recall, reasoning or architectural novelty.

### Scale diagnostics

One fixed held-out batch, averaged across four TDM layers:

| Scale | Mean routing weight |
|---|---:|
| 1 | 0.281238 |
| 2 | 0.178856 |
| 4 | 0.141563 |
| 8 | 0.138840 |
| 16 | 0.130897 |
| 32 | 0.128607 |

Initial routing was uniform1/6 with entropy1.79176. Final per-layer mean token entropy ranges1.40827–1.62918. Smaller scales receive larger average weights, but the mixture does not concentrate entirely on one scale in this probe. Routing weights alone are not output contribution magnitudes or proof of timescale semantics. Full difference RMS, projection norms, per-scale gradient norms and output RMS are in initial/final-probe.json. All measured outputs/gradients remain finite; fixed-batch gradient norm1.37831 initially and2.03775 finally.

### Fixed generations

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running towards her. Lily was scared and didn't know what to do.
> Suddenly, a big dog came running towards her. The dog barked and ran towards her. Lily was scared and ran

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very friendly and wanted to play with the dog.
> Tom said, "Hi, I am Tom. Do you want to play with me?" The dog wagged his tail and barked. Tom said, "Yes, I want to play with you!" They played with the dog and had fun. They were happy and had

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to find his toy, but he was too small. He looked everywhere, but he could not find it.
> The boy saw a big tree. He wanted to find it. He looked around and saw a big tree. He was very happy. He wanted to find the toy. He

Samples are readable but retain repetition/inconsistency. Perplexity alone does not demonstrate recall or reasoning. Full validation curve and samples are in training-report.json. Raw data and large checkpoints remain ignored by Git.
