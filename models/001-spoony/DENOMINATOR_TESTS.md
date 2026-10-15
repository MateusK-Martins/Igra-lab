# No-K recurrent-denominator normalization tests

Two fresh branches replace completed No-K FluxV's internal V RMSNorm with a second
recurrent denominator. They use the user's exact equations; all other model and
training choices are matched. No attention matrix or K projection is introduced.

## Equations and gain

Both use g=sigmoid(W_g x+b_g), u=tanh(W_V x), q=sigmoid(W_Q x), initial states 0.
A learned per-feature gamma:[384] starts at 1, exactly matching original RMSNorm
gains in count, initialization and unconstrained trainability.

```text
S[t] = g[t]*S[t-1] + u[t]
Weighted average:
    Z[t] = g[t]*Z[t-1] + 1
    branch[t] = q[t]*gamma*S[t]/(Z[t]+epsilon)
Squared retention:
    P[t] = g[t]^2*P[t-1] + 1
    branch[t] = q[t]*gamma*S[t]/sqrt(P[t]+epsilon)
output[t] = x[t] + branch[t]
```

Epsilon is fixed 1e-6, ADDED exactly as above, not a substituted clamp or recurrent
write. Both denominators have a +1 write; they do not accumulate abs(u), u², or
RMS statistics. Gamma is initialized to ones; no extra scale, clipping, gain,
router, output projection or normalization is added. The input-dependent gate
starts with zero weights/bias logit(.99), retaining original FP32 gate computation.
Q and V nonlinearities and projections are unchanged. All 12 outer RMSNorms and
SwiGLU FFNs remain; the internal RMSNorm operation is absent.

At constant scalar g=.99 and long sequences:

| Quantity | Weighted average | Squared retention |
| --- | ---: | ---: |
| Accumulated denominator state | 1/(1-g)=100 | 1/(1-g²)=50.251256 |
| Actual divisor, ignoring epsilon | 100 | sqrt(50.251256)=7.088812 |
| Interpretation | Surviving write-weight sum | Square root of surviving squared-weight sum |

The squared normalization stabilizes coefficient-weighted variance under independent,
zero-mean equal-variance writes. It is NOT an estimator of actual write variance:
write magnitudes/correlations, nonzero means and input-dependent gate/write coupling
can violate that approximation. Neither denominator uses cross-feature state RMS.
For feature-wise gates, the new divisions also change relative feature scaling,
not just one global amplitude factor. These qualifications matter when comparing
either result to RMSNorm's instantaneous input-dependent cross-feature statistic.

## Exact parameter matching and state

Both models have 12,887,040 parameters, exactly matching normalized No-K, including
six 384-feature gain vectors. Numerator/query/gate/FFN/embedding initial weights are
bit-identical between modes and match No-K (gain renamed from value_norm.weights).
No trained checkpoint is used. Original constructor draws are preserved, and gamma
creation draws no random numbers. Audit records SHA256 of every initial tensor.

One existing affine_prefix_sum handles packed [S,denominator] features. In squared
mode g² is computed in FP32 before the scan. Numerator arithmetic, scan, projections
and gain are identical; only denominator retention/divisor change as specified.
No scan optimization or per-token/head Python loop is introduced. Low-precision
state accumulation is FP32; FP64 references remain FP64. Readout division/gain use
working precision before the original BF16 output cast and Q multiplication.

Inference stores S and denominator: [B,1,384] each, 768 FP32 elements/sample/layer
(3 KiB), twice original No-K's 1.5 KiB but constant in sequence length. The generic
key_sum cache field contains denominator mass, NOT a K state; value_sum contains S.
Mode-specific cache IDs reject cross-mode/original cache reuse. Snapshot restoration
preserves both vectors and count. Strict weight loading rejects the old RMSNorm
parameter names; manifest mode/source hashes validate checkpoint compatibility.

## Training protocol

Original width 384, six layers, SwiGLU 1024, tokenizer 8192, seed 42, same pinned
TinyStories 30000 train/256 validation documents and verified exact artifacts.
Context 256, batch 8, BF16, AdamW LR3e-4,betas(.9,.95),wd.01,warmup300,cosine floor.1,
clip1,accumulation1,validation500,checkpoint1000. 9,000 updates/18,429,440 actual
target tokens each, including identical partial epoch-ending batches. Loader RNG
is independent of model initialization. The original FFN is preserved per user
choice. Training/optimizer/evaluation/generation/scan infrastructure is reused.

Benchmark/initial probes run separately and discard weights. Full training starts
fresh from seed 42 in its own process. Both recipes are frozen before training;
source/artifact/settings hashes are checked afterward. Checkpoints/events/data stay
local and ignored. Runs execute sequentially under temporary sleep/idle/lid
inhibition; thermal management remains with the user. No other stopped experimental
suite is automatically resumed.

## Verification and diagnostics

655 tests pass per branch, plus Ruff lint/format. New tests cover independent FP64
sequential values and ALL parameter/input gradients, including gamma and learned
retention, causal/future-gradient isolation, gradcheck, residual once, full/chunk/
single-token cache/snapshot/reset/constant size, cache-mode isolation, no old weight
reinterpretation, numerator equality under common input/weights, exact parameter
counts and retention closed forms at g in {0,.5,.99,1}. A coefficient-energy test
verifies the squared-weight variance identity. Zero/2049-token behavior and BF16
with FP32 gates/states/finite gradients are checked. Full-size optimizer benchmarks
verify available GPU memory before training. Existing tests remain passing.

Initial/final probes record mixer RMS, raw S, denominator, gamma distributions,
normalized-last-state RMS, gate distributions/token variation and finite gradient
norms. Write statistics include average absolute per-feature mean and lag-one
correlation, descriptive checks on zero-mean/independence assumptions. They do not
establish white-noise writes or semantic roles. State/denominator statistics sample
the last cached token; output/gate/write statistics use the whole fixed batch.
Full validation curves and inherited fixed-prompt greedy generations are preserved.
Primary loss/PPL uses final-budget state; samples use best saved checkpoint.

## Interpretation plan

Average vs squared retention is parameter/data/initialization matched. The result
compares surviving-weight averaging to coefficient-energy scaling, not a new memory
representation. Both double state relative to original No-K. Against RMSNorm,
all parameter counts/gains match, but normalization's feature coupling, write-history
conditioning and amplitude behavior differ. If squared retention matches RMSNorm,
that supports coefficient-energy control as sufficient on this recipe. If it fails,
we cannot conclude cross-feature expressiveness is necessary without considering
write means, variance, correlations and optimization scale. Learned gamma adaptation
and readout RMS help assess amplitude compensation under the fixed training budget.
Single seed/limited TinyStories exposure; no reasoning/retrieval/general superiority
claim. Previous references and incomplete branches are preserved.
