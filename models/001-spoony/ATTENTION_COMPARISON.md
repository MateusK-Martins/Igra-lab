# A16 / A4 / SMA4 controlled comparison

Lantern backbone: width 384, six layers, SwiGLU 1024, tied 8192-token embedding
(std 0.02), outer norms and residuals unchanged. Only the attention block changes.
All three use a single module/head, no RoPE/position embeddings, unrestricted
causal history, no dropout. These follow the requested equations. Historical
Lantern uses RoPE/GQA; its comparison therefore changes more than cache type.
Compression reduces actual counts: A16 10,375,680; A4 10,265,088; SMA4 10,285,968.
There is no compensatory FFN expansion.

A16/A4 use bias-free Q/K/V/output projections, SDPA scaling by sqrt(latent width),
and store only compressed K/V. SMA4 uses biased query/matrix/output projections,
a direct unconstrained 4x4 matrix per token, learned scoring vector initialized
N(0,0.5), and exactly scores u^T M q / 2. Linear projections retain PyTorch's
standard fan-in initialization. SMA responses, scores, softmax, and aggregation
are FP32 under BF16; FP64 is preserved for references. The final 4D aggregate
alone is decoded. SMA allocates [B,T,S,4] responses during training; complexity
is quadratic in tokens, with 16 cached scalars per token/layer during inference.
A16/A4 cache 32/8 scalars respectively, excluding metadata. Cache history grows
with tokens as requested. Cached execution is no-grad only. Matrix snapshots
use format 2 and distinct entries, never a disguised KV pair.

Attention construction isolates its RNG consumption, so unchanged embeddings,
FFNs and norm gains initialize identically across variants. Seed 42 and independent
loader generators are preserved. New attention weights are fresh; no warm start.
Same copied byte-identical prepared TinyStories/tokenizer artifacts, batch 8,
context/stride 256, BF16, AdamW 3e-4 / (.9,.95) / eps 1e-8 / decay .01,
clip 1, 300 warmup, cosine floor .1, 9000 updates, validation every 500,
checkpoints every 1000. Benchmark updates are discarded in a separate process.
Training manifests fingerprint inputs and sources. No dataset/checkpoint enters Git.

Run from this directory with the existing project virtual environment:
`python -m pytest -q`, `ruff check src configs scripts tests`,
`ruff format --check src configs scripts tests`,
`python -m scripts.probe_attention initial`, `python -m scripts.benchmark`,
`python -m scripts.run_baseline`, `python -m scripts.probe_attention final`.
The initial/final fixed-batch probes perform no optimizer updates. Samples follow
historical greedy generation from the best saved validation checkpoint; final
metrics use update 9000. All curves are retained in training-report.json.

507 tests passed before training. Coverage includes independent sequential
softmax/matrix-vector equations and all parameter gradients, FP64 gradcheck,
future-gradient isolation, residual once, single-token and chunked cached decoding,
snapshot/restoration, exact cache size, full-rank learned matrices, query sensitivity,
BF16 CPU/CUDA stability, long/zero inputs, exact counts and shared initialization.
