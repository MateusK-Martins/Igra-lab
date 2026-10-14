# Controlled FluxV 14M ablations

Reference: 29890bc8a7327667102eceb823317a009b96d32e. Historical V-only result:
13,774,080 parameters, loss 2.1738922596, PPL 8.792440, 9,000 updates,
18,429,440 actual target tokens (partial epoch-ending batches explain the small
shortfall from nominal 18,432,000). Historical artifacts are preserved.

Every branch starts fresh from seed42 with the original tokenizer and exact
train/validation stores (30,000 / 256 TinyStories documents). Dataset fingerprints
and all resolved training settings are checked against the historical manifest.
Context256, batch8, BF16, AdamW LR3e-4, betas(.9,.95), wd.01, warmup300, cosine
floor.1, clip1, accumulation1, validation500, checkpoint1000, 9,000 updates.
The existing training, optimizer, loader, scan and generation scripts are reused.
No QA stage is part of this experiment. No original branch/artifact is modified.

| Model | K/V normalization | Readout | Parameters |
| --- | --- | --- | ---: |
| Reference | Global RMSNorm | Element-wise sigmoid Q times V | 13,774,080 |
| A | Six independent 64-feature RMS statistics | Original element-wise | 13,774,080 |
| B | Same head normalization as A | Cyclic factored matrix interactions | 13,774,080 |
| C | Original global RMSNorm | Identical matrix interactions | 13,774,080 |
| D | Original global RMSNorm | Residual shared head-axis 6→24→6 GELU MLP | 13,775,988 |
| E | No heads; RMS-normalized 128-feature matrix readout | Single decayed 64×128 matrix | 14,211,456 |

A–D retain full-width cumulative K, token-dependent V decay, original FP32 vector
states, existing affine scan, outer norms, FFN, embedding and residual wrappers.
Head normalization keeps 384 learned gains for each norm and epsilon1e-6; only
normalization domains change. Matrix B/C uses q=sigmoid(Wq x), with 288 outputs,
P:[6,64,48], W:[6,48,64], u_h=v_h P_h, z_h=v_(h+1 mod6)*((q_h*u_h)W_h),
concatenated in cyclic-pair order. No explicit temporary matrices are allocated.
P and W are initialized independently Normal(0,1/sqrt(fan_in)); fixed output
scaling is 1. Reduced Q saves36,864 parameters/layer, exactly offset by P+W.

Initialization draws the original full Q before replacing it inside a CPU RNG
fork. Extra P/W and MLP initialization also uses forks, preserving the random
stream and unchanged parameter initial values. B and C start with identical
weights. D's MLP includes biases (318 parameters/layer; 1,908 total); final
projection weight/bias are zero, making its initial readout exactly the baseline.
No trainable extra gains, routing, output projections or head semantics claims.

E is a separate architecture, not part of the 2×2 normalization/readout ablation.
It retains full-width bias-free Q/K/V, applies sigmoid/tanh/sigmoid respectively,
then bias-free down-projections to64/64/128. Its write is
(k_down(tanh(Kx))/sqrt64) outer v_down(sigmoid(Vx)). Gate row/column projections
384→64 and384→128 are FP32 sigmoid; zero weights, biases logit(sqrt(.99)). Their
outer product gates the previous matrix element-wise, before the current write.
Q reads the updated matrix with an additional fixed1/sqrt64 query scale. The
128-dimensional read gets trainable global RMSNorm(eps1e-6), then bias-free128→384.
There is no separate cumulative K or V. Outer FFN/norms are unchanged. Count is
437,376 above baseline (+3.175%); no artificial capacity padding.

E uses the original affine_prefix_sum on flattened matrix states inside chunks
of16 tokens, with non-reentrant PyTorch activation checkpointing. Chunk boundaries
are sequential, gradients cross them without detach. Within each chunk, the scan
remains parallel. This changes execution cost, not recurrence/context/exposure;
no scan optimization is attempted. Training materializes only local [B,16,64,128]
matrices; inference stores a single [B,64,128] FP32 matrix per layer in its own
MatrixStateEntry. Vector caches cannot be reinterpreted as matrix caches. The
matrix uses8192 state elements/sample/layer vs768 for two vectors (10.67×).

Validation curves and fixed greedy prompts use the original procedures. Reports
show final-budget loss/PPL as primary; inherited run_baseline also records the best
saved checkpoint and generations from that checkpoint, explicitly labeled.
Initial/final fixed-held-out-batch probes measure mixer output RMS/variance,
per-head state/output amplitudes for A–D, gate distributions/token variation,
P/W norms, D MLP relative change, finite gradients and total gradient norm. E
records matrix norm, sample0 singular values/entropy effective rank and query
sensitivity on fixed memory. These diagnostics are not reasoning/retrieval tests.
Training itself retains the existing finite-gradient check at every BF16 update.

Commands: python -m scripts.probe_variants initial; python -m scripts.benchmark;
python -m scripts.run_baseline; python -m scripts.probe_variants final. Use the
model .venv; benchmarking/probes are separate processes so they never alter fresh
training initialization or loader RNG. Data and checkpoints remain ignored.

Preflight complete: A–D full suites636 tests each; E623 tests, including full-size
CUDA memory/backward. Ruff passes. Every data/settings audit matches the historical
manifest. Historical activation/gradient probes are reproduced from a temporary
checkout of the exact reference commit, loading its preserved final weights only
for its final probe. Benchmark and probe weights are always discarded before fresh
training. E also forks extra-projection initialization to preserve unchanged outer
FFN and later-layer seed values.
