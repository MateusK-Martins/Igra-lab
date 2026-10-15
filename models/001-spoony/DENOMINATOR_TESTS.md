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

## Completed comparison

Both fresh runs reached 9,000 updates and 18,429,440 target tokens. Post-training
checks verified source fingerprints, identical data/settings, initial tensor hashes
and finite probes. Both have exactly the same 12,887,040 trainable parameters as
normalized No-K, including learned gamma. No warm-start or post-result tuning.

| Model | Parameters | Final loss | Perplexity | Measured min | Tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FluxV | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 46,589 | 1.044 |
| E recurrent matrix memory | 14,211,456 | 2.039836 | 7.6893 | 35.04 | 7,792 | 1.044 |
| Lantern attention | 12,440,064 | 2.018320 | 7.5257 | 7.19 | 42,727 | 0.748 |
| No-K signed writes | 12,887,040 | 2.123176 | 8.3576 | 6.84 | 44,938 | 0.940 |
| diagonal | 12,002,304 | 2.434742 | 11.4129 | 4.97 | 61,864 | 0.781 |
| no-norm | 12,884,736 | 2.743455 | 15.5406 | 5.84 | 52,637 | 0.903 |
| weighted-average | 12,887,040 | 2.172995 | 8.7846 | 10.36 | 29,647 | 1.294 |
| squared-retention | 12,887,040 | 2.112602 | 8.2697 | 10.62 | 28,926 | 1.285 |

E timing is its resumed 8000-update phase, not the full 9000 updates. All losses
are final-budget values. Other reference timings are full historical sessions.

### Controlled loss contrasts

Positive differences are worse:

- average_minus_RMSNorm: +0.049819 nats/token.
- squared_minus_RMSNorm: -0.010574 nats/token.
- squared_minus_average: -0.060393 nats/token.

Squared retention has the lower final loss of the two in this seed. This is a matched
parameter/data/initialization comparison of the requested denominator mechanisms,
not a statistical significance result.

### What this says about normalization

Weighted averaging divides by surviving write mass and bounds the un-gained
normalized state by 1 in absolute value. Signed cancellation can make this average small,
and the division does not restore the remaining signal amplitude. Squared retention divides by coefficient-energy
scale, preserving more amplitude for independent zero-mean writes. Its variance
argument does not hold automatically for learned nonzero-mean, correlated writes
and input-dependent retention. The diagnostics below test those assumptions only
descriptively on a fixed batch.

RMSNorm instead uses the current state's cross-feature second moment (including
mean offsets), and does not estimate historical surviving mass. Because its divisor
is common across features, it preserves feature ratios up to learned gains; the
feature-wise historical denominators can alter those ratios based on each feature's
retention history. These are distinct inductive biases, not equivalent variance
estimators.

Squared retention approaches and slightly surpasses RMSNorm here: loss improves
by 0.010574 nats/token, about 1.05% in perplexity. It beats weighted averaging by
0.060393 nats/token (about 5.86% perplexity). This supports the user's hypothesis
that coefficient-energy scaling is more useful than true historical averaging
for these signed writes at this budget. It provides a viable replacement without
an instantaneous cross-feature RMS statistic on this recipe.

The measured amplitudes are consistent with that account: final mixer RMS is
0.5792 for squared retention, 0.3875 for averaging, and 0.4990 for RMSNorm No-K.
Squared retention's gain RMS remains near 1.0175; averaging's reaches 1.1087.
Averaging began with much smaller mixer RMS (0.0538), while squared retention
began at 1.4367. Training adapts writes/gates as well as gains, so these figures
do not isolate amplitude as the causal reason for the quality gap.

Learned writes are not perfect independent zero-mean noise: sampled per-feature
mean offsets and lag-one correlations remain nonzero. Squared retention generally
has smaller offsets/correlations than averaging on this probe, but their learned
trajectories differ. Thus this is not proof RMSNorm's ONLY role is variance control,
or that cross-feature coupling is always unnecessary. The extra denominator also
adds history-dependent feature scaling and doubles recurrent state versus RMSNorm.
Repeat-seed evidence is needed before treating the small RMSNorm advantage reversal
as robust; the paired average/squared difference is still a single-seed observation.

The cost matters: squared retention took 10.62 min versus 6.84 min for RMSNorm No-K,
with 3 KiB versus 1.5 KiB fixed inference state and 1.285 versus 0.940 GiB peak training
CUDA allocation. Historical sessions limit exact speed attribution, but this
implementation trades additional state/scan work for the modest quality change.
These results do not establish attention/retrieval behavior; no query-key matrix
or token-growing memory is involved.

### Readout, gain, denominator and gradient diagnostics

Values average across six layers of one fixed validation-batch probe. Mixer/gates/
writes use full batch/time; raw state, denominator and normalized readout use last
cached token. Gradient norms are un-clipped probe norms. All are finite; training
retained the existing clip 1/nonfinite-gradient guard at every BF16 update.

| Model | Mixer RMS initial → final | Final raw S RMS | Final normalized-state RMS | Final gamma RMS | Gradient norm initial → final |
| --- | --- | ---: | ---: | ---: | --- |
| weighted-average | 0.0538 → 0.3875 | 10.6647 | 0.4953 | 1.1087 | 1.4117 → 3.0990 |
| squared-retention | 1.4367 → 0.5792 | 4.3506 | 1.1680 | 1.0175 | 3.4505 → 1.8740 |

Per-layer denominator, gain and write diagnostics:

| Model/layer | Denominator-state mean | Gamma mean/min/max | Mean absolute feature write mean | Mean lag-one write correlation | Mean retention | Token variation std |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| weighted-average/body.1.branch | 18.5301 | 1.0678/0.9776/1.2333 | 0.1917 | -0.0129 | 0.85544 | 0.21489 |
| weighted-average/body.5.branch | 18.6983 | 1.0807/0.9892/1.2617 | 0.2019 | 0.0900 | 0.80484 | 0.22301 |
| weighted-average/body.9.branch | 17.1973 | 1.1111/0.9948/1.2469 | 0.1466 | 0.1808 | 0.79580 | 0.23065 |
| weighted-average/body.13.branch | 16.4903 | 1.1283/1.0130/1.2690 | 0.1413 | 0.3395 | 0.80431 | 0.23836 |
| weighted-average/body.17.branch | 14.3886 | 1.1331/1.0371/1.2417 | 0.1542 | 0.4502 | 0.80097 | 0.23503 |
| weighted-average/body.21.branch | 15.6953 | 1.1259/1.0324/1.2447 | 0.1642 | 0.5025 | 0.84219 | 0.20063 |
| squared-retention/body.1.branch | 3.6734 | 1.0341/0.9362/1.1883 | 0.0925 | -0.0242 | 0.76136 | 0.28207 |
| squared-retention/body.5.branch | 5.0739 | 1.0362/0.9532/1.2405 | 0.1015 | 0.0304 | 0.80037 | 0.28126 |
| squared-retention/body.9.branch | 10.4191 | 0.9938/0.9351/1.1741 | 0.1148 | 0.1565 | 0.86884 | 0.19115 |
| squared-retention/body.13.branch | 10.6113 | 1.0072/0.9435/1.1968 | 0.1253 | 0.1866 | 0.85170 | 0.19842 |
| squared-retention/body.17.branch | 8.2901 | 1.0148/0.9292/1.1911 | 0.1406 | 0.2174 | 0.82264 | 0.21191 |
| squared-retention/body.21.branch | 5.4489 | 1.0125/0.9462/1.1812 | 0.1536 | 0.3103 | 0.81192 | 0.20376 |

Denominator state means are Z for average and P for squared retention; P is square-
rooted before division, so those columns are not directly comparable divisors.
Gamma values show learned static amplitude compensation, not instantaneous feature
normalization. Lag-one correlation does not characterize all temporal dependence,
and fixed-batch feature means do not prove stationary population means. These
diagnostics qualify the independent zero-mean variance argument rather than
establish causal explanations. Complete distributions/per-parameter gradients,
curves and samples remain in JSON. Both inference states are 3 KiB/sample/layer
versus 1.5 KiB for RMSNorm No-K; training throughput/VRAM also differ.

### Fixed-prompt generations

Inherited greedy procedure using best saved checkpoint, 64 new tokens max. Primary
metrics above use final-budget state. These are illustrative stories, not QA tests.

#### weighted-average — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very friendly and Lily wanted to play with it.
> Lily asked her mommy, "Can I play with you?" The boy said, "No, I don't want to play with you." Lily felt

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and wanted to play with the dog. He ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog barked and wagged his tail.
> The dog wagged its tail and wagged its tail. It wagged its tail and wagged its tail. It wagged its tail and

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his friends. He wanted to play with his friends.
> The little boy went to the park with his friends. They played together and had fun. They played together and had fun. The boy was happy to have a new friend.

#### squared-retention — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very friendly and Lily wanted to play with her.
> Lily ran to the dog and said, "Hi, I'm Lily. Do you want to play with me?" The dog wagged its tail and wagged

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and played with Tom. Tom was not scared of the dog anymore. He wanted to play with Tom and his friends.
> Tom's mom saw him and said, "Tom, you should not touch the dog. It is not safe. It is not safe. It is just a toy. It is a

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He looked everywhere for it but he couldn't find it.
> Then, he saw a big dog. The dog was very friendly and he wanted to play with it. The dog ran to the dog and started to play with it. The dog was very happy and played with the dog.
> The
