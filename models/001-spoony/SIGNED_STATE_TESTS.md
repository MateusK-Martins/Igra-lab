# Signed-state decay and readout-normalization tests

Requested after completed No-K: run the simple diagonal-state control and a fresh
No-K variant removing only the internal RMSNorm after V. Two-K and mass-normalized
full training remain stopped; this request resumes neither of them.

## Exact equations

All writes are signed tanh; Q=sigmoid(W_Q x). Outer norms/FFNs/residuals unchanged.

```text
No-K reference: S[t] = g(x[t])*S[t-1] + tanh(W_V x[t])
               branch[t] = Q[t]*RMSNorm(S[t])
Simple diagonal: S[t] = d*S[t-1] + tanh(W_V x[t])
                 branch[t] = Q[t]*RMSNorm(S[t])
No-K/no-V-norm: S[t] = g(x[t])*S[t-1] + tanh(W_V x[t])
                branch[t] = Q[t]*S[t]
output[t] = x[t] + branch[t]
```

The token-dependent g is the established FP32 sigmoid gate with zero initial
weights and bias logit(.99). Fixed d is a learned per-feature sigmoid vector,
initially .99, not token-dependent. The no-norm run preserves g and every write/
readout projection. It removes six384-feature RMSNorm gain vectors (2304 weights),
not outer model RMSNorms. No output clipping, new gain, rescaling, attention,
FFN change or scan replacement is added. Default SignedState behavior stays
normalized; a boolean normalize_state setting selects the new variant.

| Model | Parameters | Internal V RMSNorm | Retention |
| --- | ---: | --- | --- |
| Completed No-K reference | 12,887,040 | Yes | Token-dependent |
| Simple diagonal | 12,002,304 | Yes | Learned fixed diagonal |
| New No-K/no-V-norm | 12,884,736 | No | Token-dependent |

All maintain one384-feature FP32 state per layer/sample:1.5KiB, constant in context.
The generic K cache tensor has zero elements. Both normalizations share the same
raw recurrent-state semantics. Strict model loading rejects incompatible norm
parameter sets; all training starts from scratch, not from completed weights.

## Controlled protocol

Original width384, six layers, SwiGLU1024, tokenizer8192, seed42, TinyStories30000
train/256 validation documents, context256,batch8,BF16,9000updates/18429440actual
target tokens. Same optimizer/scheduler/clipping/loader order/validation intervals
as original FluxV/Lantern and completed No-K. User explicitly preserved FFN width,
accepting parameter differences. Fingerprints/settings and unchanged initialization
are audited against original; norm deletion introduces no RNG shift. Both tests
retain exact initial values of Q/V/gate/outer weights. No trained weight warm start.

The diagonal recipe was already frozen/tested at2453aee and never trained in the
stopped suite. Its source/config remain unchanged. New no-norm starts from the
No-K frozen source at45a5482; its runtime option and649-test suite cover independent
sequential values/all parameter gradients, causal/future isolation, full/chunked/
single-token cache/snapshot/reset, zero/long sequences, BF16/FP32 state, exact
counts, matching unchanged initialization and preservation of all12 outer RMSNorms.
Diagonal's existing638-test suite and preflight passed; no source change occurred.
Ruff passes. Full-size optimizer benchmark checks memory before full training.

Both fresh runs execute sequentially under temporary sleep/idle/lid inhibition.
The user manages temperature; no thermal throttling/guard is introduced. Dataset,
events and checkpoints remain local/ignored. Reports include final loss/PPL, full
curves, fixed-prompt generations, throughput/VRAM and initial/final gradient/state/
readout/retention probes. Historical timing is from earlier sessions.

## Interpretation

Simple diagonal vs completed normalized No-K mainly tests input-dependent retention,
with gate-projection capacity as a confound. No-norm vs normalized No-K closely
isolates readout RMSNorm (only2304 parameter difference), allowing historical state
amplitude to affect the residual directly. Readout/state RMS and gradient probes
help characterize that change; lower loss alone cannot identify its causal benefit.
These do not compare all recurrence/normalization families. Single seed, limited
TinyStories exposure; no reasoning/retrieval/general-superiority claim.

## Completed signed-state comparison

Both new runs completed 9,000 updates/18,429,440 target tokens from scratch.
Artifact/settings and frozen source hashes match preflight audits. Their starting
weights were not taken from the completed normalized No-K model. All 18 validation
points, fixed generations and probes are preserved.

| Model | Parameters | Final loss | Perplexity | Measured min | Tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FluxV | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 46,589 | 1.044 |
| E recurrent matrix memory | 14,211,456 | 2.039836 | 7.6893 | 35.04 | 7,792 | 1.044 |
| Lantern attention | 12,440,064 | 2.018320 | 7.5257 | 7.19 | 42,727 | 0.748 |
| 4-K independent Q | 17,331,456 | 2.135759 | 8.4635 | 15.78 | 19,462 | 1.811 |
| No-K signed writes | 12,887,040 | 2.123176 | 8.3576 | 6.84 | 44,938 | 0.940 |
| diagonal | 12,002,304 | 2.434742 | 11.4129 | 4.97 | 61,864 | 0.781 |
| no-norm | 12,884,736 | 2.743455 | 15.5406 | 5.84 | 52,637 | 0.903 |

E matrix-memory timing above covers its resumed 8000-update phase, not its full 9000 updates. All primary losses use final 9000-update state. Other reference timings are full historical runs.

### Mechanistic interpretation

**Fixed versus input-dependent retention:** Diagonal-state minus normalized No-K
loss is +0.311566 nats/token. They share signed tanh
writes, Q gating, V RMSNorm, outer blocks and unchanged initial tensors. Token
retention can selectively forget based on the current input; a fixed learned
diagonal can only choose per-feature timescales. Gate capacity also differs by
884,736 parameters, so the result does not isolate adaptation from gate capacity.

**Removing internal V RMSNorm:** No-norm minus normalized No-K loss is
+0.620279 nats/token, with only 2,304 fewer parameters.
The recurrence and Q/V/decay/outer weights start identically. This is the cleaner
readout-normalization comparison: RMSNorm removes state-amplitude dependence and
provides learned gains, whereas direct Q*S exposes accumulated amplitude to the
residual and FFN. All outer norms remain. Observe readout/state RMS below rather
than assume normalization is inherently harmful or helpful.

The diagonal and no-norm models are different axes of the signed-state family,
not a complete 2×2 experiment: fixed diagonal WITHOUT V norm was tested for
mathematical correctness but not trained. We cannot infer their interaction from
these three completed training configurations.

No-K's earlier advantage over cascaded FluxV remains evidence that direct signed
writes are viable at this budget, not proof K is universally unnecessary. E and
Lantern use different memory representations. Comparisons remain single-seed,
not demonstrations of retrieval, reasoning, semantic feature roles or general
architecture superiority. Two-K and signed-mass full training remain stopped.

For this recipe, the completed signed-state comparisons favor retaining BOTH
token-dependent decay and internal V RMSNorm. The fixed-diagonal and normalized
token-gate models had identical initial fixed-batch/full validation loss, making
the later gap unlikely to arise from mismatched starting activations. Gate capacity
still differs. Removing V RMSNorm leaves all training numerically finite but
substantially worsens learning within this budget. Its final mean mixer RMS is
8.5661 versus 0.4990 with RMSNorm (~17.2×), consistent with excessive memory-branch
amplitude affecting the residual/FFN balance. This is a diagnostic association,
not proof of the optimization mechanism or a verdict on every unnormalized design.

### Activation, state and gradient evidence

The table averages mixer/state/retention RMS or means over six layers of one fixed
held-out batch. State RMS samples the last cached token; mixer output/gates sample
the full batch. Gradient norms are un-clipped probe norms. All probes are finite;
training retains its nonfinite-gradient guard and clip 1.

| Model | Mixer RMS initial → final | Final raw state RMS | Gradient norm initial → final | Final mean retention | Token variation std |
| --- | --- | ---: | --- | ---: | ---: |
| Normalized No-K | 0.5169 → 0.4990 | 8.2596 | 3.1325 → 1.7139 | 0.90128 | 0.13973 |
| diagonal | 0.5169 → 0.4959 | 17.5257 | 3.1270 → 1.9292 | 0.98669 | 0.00000 |
| no-norm | 12.1294 → 8.5661 | 21.0288 | 5.5256 → 2.0951 | 0.81968 | 0.20564 |

Per-layer retention distributions, gradients and state statistics are retained in
the full probes. Zero token variation for diagonal decay is architectural; nonzero
variation for the projected gates verifies input dependence on this batch, not
semantic forgetting behavior. Identical FP32 state/cache layout does not imply
identical learned contents after different training trajectories.

### Fixed-prompt greedy generations

Inherited best-saved-checkpoint procedure, 64 new tokens max. Primary loss/PPL uses
each final-budget checkpoint. Samples are illustrative stories, not QA evaluations.

#### diagonal — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was scared and ran away. Lily was scared and ran away.
> Lily's mom saw her and said, "Don't worry, Lily. The dog is just a big dog. It's just a big

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy. He wanted to play with the dog, but he was too small. He wanted to play with the dog, but he was too small to play with.
> The big dog saw the big dog. The dog was scared of the dog and wanted to play with the dog. The dog ran away and ran

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the boy and his toy. He wanted to play with it. He looked for a little girl who was sad. She had an idea. She had an idea. She took the boy to the boy and gave it to the boy. The boy was very happy. He said

#### no-norm — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, scary bird. She was very sad and wanted to play with it.
> Lily asked her mom, "Can I play with you?" Her mom said, "Yes, I can help you."
> Her mom smiled and said, "

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and he wanted to play with it.
> The man said, "I'm sorry, little bird. I am sorry. I want to play with you."
> The bird said, "No, I can help you. I will help you."
> The bird said, "Yes, I can help you

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt a little bit scared. He wanted to play with it, but he was too small. He wanted to play with it, but he was too small.
> He ran to the dog and tried to catch it. He saw a big tree with a big tree. He wanted to catch it, but he was too fast
