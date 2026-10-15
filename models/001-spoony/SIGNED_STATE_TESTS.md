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
