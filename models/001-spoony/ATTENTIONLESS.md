> **Ember:** this model now lives on `model/001-spoony-ember`. Historical
> recipe commits and reports retain their original branch identities.

> **Current experiments:** [SIGMOID_NORM.md](SIGMOID_NORM.md) tracks the
> sigmoid Q/V runs. The original unbounded-Q/V experiment below is historical.

# Cumulative mechanism — codename pending

Branch: `codex/attentionless`, based on Lantern. The configured model replaces
all six GQA branches with CumulativeMixer; surrounding RMSNorm, SwiGLU, tied
embedding/head and dimensions remain unchanged. No RoPE, attention matrix,
softmax, window mask or output projection is used in the replacement.

For input [B,T,D], the implemented branch is:

```python
k = tanh(k_proj(x))
q = q_proj(x)
v = v_proj(x)
k_sum = cumsum(k, dim=1)
k_normalized = RMSNorm(k_sum)
v_sum = cumsum(v * k_normalized, dim=1)
branch = q * RMSNorm(v_sum)
output = x + branch
```

Both norms are independent, trainable feature-wise RMSNorm with epsilon 1e-6.
The existing ResidualImpl adds x exactly once; build() returns only the branch.
BF16/FP16 projected values accumulate in FP32, preserving float64 for reference
checks. There is no division by token count or cross-token normalization.

Uncached training resets sums at each sampled 256-token window, matching
Lantern's training context. Cached inference carries the raw (unnormalized)
key sum and raw value-product sum, [B,1,D] each, through external StateEntry
objects in CacheStorage.states. Their memory does not grow with sequence length.
Cached execution requires no_grad, supports CPU/CUDA state storage and snapshots,
and does not truncate history at 256 tokens. A future long-context comparison
must account for that difference from Lantern's windowed inference. Existing KV
entries and snapshots remain supported for the preserved GQA code/tests.

The direct replacement has 12,887,040 parameters with vocabulary 8192, versus
Lantern's 12,440,064 (+446,976, +3.59%). We preserve the requested mechanism and
surrounding dimensions rather than changing SwiGLU to force a parameter match.
This is an equal-token direct-swap experiment, not an exact parameter-matched
architecture comparison.

The same prepared TinyStories tokenizer/splits are copied locally from Lantern,
with artifact hashes verified before training. The seed, data order, batch size,
precision, optimizer, 9,000-update schedule and validation settings are preserved.
Fresh outputs use `results/pretraining/attentionless-tinystories-v1/`.
Run `.venv/bin/python -m scripts.benchmark`, then
`.venv/bin/python -m scripts.run_baseline` from this model directory.

Tests include a sequential scalar-time reference for outputs and all gradients,
future-token value/gradient isolation, exactly one residual, finite zero-input
backpropagation, single-token/chunked/full equivalence, snapshot continuation,
constant-size state, cache rejection without mutation, definition validation,
and CUDA BF16/FP16 gradients with CPU-resident FP32 running sums.
No training result is implied by the inherited Lantern report in CODENAME.md.

## Completed equal-token run — 2026-10-13

Recipe commit: `09782df`. All 495 tests passed; Ruff lint and formatting passed.
The fresh seed-42 run completed 9,000 updates and 18,429,440 target tokens in
290.50 seconds (4 minutes 50 seconds), including validation and saving. Initial
validation loss was 9.090187; final loss 2.659388, perplexity 14.287542. Final
checkpoint step-00009000.pt is best among nine saved checkpoints. It was loaded
for fixed greedy generations, and its settings match the saved manifest.

Lantern used 431.33 seconds for exactly the same consumed target tokens, with
final validation loss 2.018320 and perplexity 7.525675. This mechanism was about
1.48 times as fast overall (32.65% less elapsed training time), but learned less
effectively under this fixed recipe. The cumulative model has 3.59% more
parameters. This does not establish an architecture ranking: one seed, one
normalization choice, one untuned recipe and no equal-time follow-up were tested.

Fixed-prompt generations are readable but repetitive and inconsistent; the
complete samples and curve are in
[training-report.json](results/pretraining/attentionless-tinystories-v1/training-report.json).
See [comparison.json](results/pretraining/attentionless-tinystories-v1/comparison.json)
and the saved manifest for the exact comparison and artifact hashes. Prepared
artifacts were checked against Lantern byte for byte, and both runs consumed
the same number of target tokens. All weights and raw/prepared data remain
ignored and local to this worktree. No additional training run has been started.
