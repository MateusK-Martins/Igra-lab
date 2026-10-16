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

## Completed results

All runs completed 9,000 fresh updates / 18,429,440 target tokens. Dataset and tokenizer fingerprints match historical Lantern. Initial shared-backbone weight hashes match across A16/A4/SMA4. Sources remained identical to each frozen training manifest.

| Model | Parameters | Loss | Perplexity | Minutes | Target tokens/s | Peak GiB | Cache scalars/token/layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A16 | 10,375,680 | 2.271129 | 9.690332 | 3.69 | 83,307 | 0.560 | 32 |
| A4 | 10,265,088 | 2.488885 | 12.047835 | 3.91 | 78,573 | 0.571 | 8 |
| SMA4 | 10,285,968 | 2.677201 | 14.544328 | 4.58 | 67,063 | 0.610 | 16 |
| Lantern (historical) | 12,440,064 | 2.018320 | 7.525675 | 7.19 | 42,727 | 0.748 | 192 |

The backbone remains full width 384 and FFN width 1024. The approximately 14M pipeline does not make the compressed variants 14M models: removing most attention parameters yields the counts above. No padding or FFN changes were made.

A16 versus A4 isolates latent width with the same conventional attention mechanism. SMA4 versus A4 tests query-conditioned content at the same query/readout width, but with 16 versus 8 cached scalars per token, 20,880 additional parameters, and the explicitly requested biases. SMA4 versus A16 uses half the cache but a narrower query/readout. These are useful trade-off comparisons, not fully parameter/cache-matched ablations.

SMA4 scores satisfy uᵀ Mᵢ q = (Mᵢᵀu)ᵀq. Its selection remains a four-dimensional effective-key dot product. The distinct operation is its content: Mᵢ q varies with the query. The implementation still stores the full, unconstrained 4×4 matrix and computes responses before scores; it does not replace it with KV attention.

Observed SMA4 minus A4 loss: +0.188316; SMA4 minus A16: +0.406072. The matrix response did not improve on A4 in this run. This does not establish a general architectural advantage or demonstrate reasoning/retrieval. All comparisons are single-seed and limited to the same TinyStories budget.

Historical Lantern is a contextual reference, not a clean cache-type ablation: it uses GQA, RoPE, more attention parameters, and a different original RNG consumption pattern. All three current variants omit positional rotation to follow the supplied equations.

## Stability and samples

Initial and final fixed-batch logits and gradients were finite for all variants; BF16 training completed without nonfinite-gradient failures. Diagnostic gradient norms are measured before the configured clipping at 1.0. Readout RMS, attention entropy, gradient norms, BF16 cached-logit differences, and SMA matrix singular values are recorded in initial-probe.json and final-probe.json. FP64 cache equivalence is tested separately, so BF16 kernel-rounding differences are not mistaken for changed causal semantics.

### A16

Gradient norm before clipping: 1.487765 initially → 4.208115 finally.

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball. It was so pretty and shiny. She wanted to play with it, but it was too high for her to reach.
Lily's mom saw her and said, "Mommy, can I play with the ball?" Her

### A4

Gradient norm before clipping: 1.479060 initially → 4.683069 finally.

Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she saw a big, shiny toy car. She wanted to play with it, but it was too high. 
Lily's mom said, "I can't find it. It's too high." 
Lily was sad and didn't

### SMA4

Gradient norm before clipping: 2.023092 initially → 3.756281 finally.

Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around in the park. One day, she saw a big, shiny rock in the sky. She was so excited to see what was. She was so excited!
Lily was so excited! She asked her mom if she could go to the park. Her mom said yes

Samples use the same fixed prompts and greedy decoding from the best saved validation checkpoint, following the existing pipeline. Final table metrics use update 9000. All prompts and full validation curves are in training-report.json.

## Reproducibility

Each branch contains the same interchangeable implementation and mechanism tests, with only its mode/output paths changed. Each branch passed 507 tests and Ruff lint/format checks before training. Runtime manifests record exact recipe commits, sources, environment, artifact hashes and settings. Checkpoints and raw data remain local and ignored. comparison.json contains cross-run numeric results; per-run reports/probes remain under results/pretraining/<mode>-14m.

## Trained-cache cross-check

The final trained checkpoints also passed full-sequence versus 7-token cached
chunks on a fixed 21-token prompt in FP32 (maximum logit differences below
1.2e-5) and FP64 (below 3.2e-14). See trained-cache-equivalence.json.
The larger BF16 differences therefore reflect low-precision execution in this
probe; cached inference should still use suitable precision tolerances rather
than expect bitwise equality across chunk shapes.

SMA4 additionally couples query scale to response magnitude: multiplying q by c
multiplies both selection logits and every Mq response by c. This differs from
A4, where query scaling changes scores but not stored values. That coupling is
one possible optimization issue, not an experimentally isolated explanation.
Initial SMA output RMS was also higher than A4 (see probes), and the requested
SMA biases differ from the conventional attention equations. These results test
the specified mechanisms rather than a scale/bias-matched auxiliary control.
