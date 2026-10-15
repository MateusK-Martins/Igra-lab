# No-K signed-state write/readout experiments

Three independent fresh branches start from the normalized No-K recipe at 45a5482,
not from trained weights, denominator normalization or the no-V-norm variant.
Original No-K reference: 12,887,040 parameters, loss 2.1231759831/PPL8.3576391014.

## Shared mechanism and exact ablations

All retain token-dependent g=sigmoid(W_g x+b_g), signed tanh V writes, trainable
internal V RMSNorm and outer model blocks. Retention uses the existing gate's
zero initial weights and bias logit(.99). Gate bias addition/sigmoid and recurrent
accumulation remain FP32 under BF16. Original Q/V projections are bias-free.

### Independent signed-write magnitude gate

```text
i[t] = sigmoid(W_I x[t]+b_I)
S[t] = g[t]*S[t-1] + i[t]*tanh(W_V x[t])
branch[t] = sigmoid(W_Q x[t])*RMSNorm(S[t])
output[t] = x[t]+branch[t]
```

W_I:[384,384] and b_I:[384] are independent of retention gate parameters. Both
start at 0, giving i=.5. A constant scalar write factor largely cancels under
RMSNorm (apart from epsilon), retaining a comparable initial readout scale while
avoiding an initially saturated input gate. Reuse the existing projected sigmoid
gate runtime; it controls the write, never multiplies the previous state. A test
closes writes while retaining old state to verify independence. Learned input gates
can suppress new writes without requiring the retention gate to forget old state.

### Signed readout query

```text
S[t] = g[t]*S[t-1] + tanh(W_V x[t])
branch[t] = tanh(W_Q x[t])*RMSNorm(S[t])
output[t] = x[t]+branch[t]
```

Only the Q activation changes. Baseline memory is already signed, so sigmoid Q
already permits negative outputs when S is negative; tanh Q adds input-dependent
SIGN REVERSAL relative to the stored feature, rather than introducing the first
negative contributions. No projection, state or scan parameters are added.

### Direct low-rank state readout mixing

```text
S[t] = g[t]*S[t-1] + tanh(W_V x[t])
r[t] = U V RMSNorm(S[t])
branch[t] = sigmoid(W_Q x[t])*r[t]
output[t] = x[t]+branch[t]
```

D=384,rank r=32,V:[32,384],U:[384,32], both bias-free and independent per layer.
Production uses two factored Linear operations, not a full [384,384] temporary
matrix. No nonlinearity between factors, no identity/residual readout path, router,
softmax or matrix memory. This is the user's DIRECT low-rank formula: it introduces
a rank 32 bottleneck as well as learned cross-feature mixing. A failed result would
not establish that all mixing is harmful, since identity/full-rank readout capacity
is also removed. The full 384-feature recurrent state is retained, not compressed.

Each factor starts Normal(0,1/sqrt(fan_in)) to keep expected initial output second
moment comparable to normalized input. This does NOT approximate identity mapping.
Extra initialization runs inside a CPU RNG fork, preserving every original Q/V/gate/
RMSNorm/embedding/FFN tensor and downstream RNG stream. Input-gate zero initialization
also preserves the stream. All common initial tensors are audited against No-K and
between the three variants. No trained-weight warm start or post-result tuning.

## Parameter, memory and protocol controls

| Model | Parameters | Added per layer | Inference state elements/sample/layer |
| --- | ---: | ---: | ---: |
| Normalized No-K reference | 12,887,040 | — | 384 |
| Independent write gate | 13,774,080 | 384²+384=147840 | 384 |
| Signed readout | 12,887,040 | 0 | 384 |
| Rank 32 readout | 13,034,496 | 2×384×32=24576 | 384 |

The write-gate model exactly matches original cascaded FluxV's parameter count
without padding. FFN 1024, width 384 and six layers remain fixed for ALL branches,
as requested in the preceding suite. Counts are within the established 13–14M range;
no capacity redistribution or environment redesign is used. Each stores one 384-
feature FP32 S vector (1.5 KiB/sample/layer), constant in sequence length; generic
key cache has zero elements. Low-rank readout has no persistent matrix state.
Mode-specific cache IDs and frozen manifest/source settings protect compatibility.

Same seed 42, tokenizer 8192, pinned TinyStories 30000 train/256 validation documents,
exact prepared-artifact fingerprints, context256,batch8,BF16,9000 updates and
18429440 actual target tokens (matching partial epoch-ending batches). Same AdamW
LR3e-4,betas(.9,.95),wd.01,warmup300,cosine floor.1,clip1,accumulation1,
validation500/checkpoint1000. Loader RNG is independent of model initialization.
Existing training/optimizer/scheduler/scan/evaluation/generation paths are reused.
One unchanged vector affine scan per layer; no token loops or scan optimization.

Benchmarks and initial probes discard weights in separate processes. Full training
starts fresh from seed 42. Runs are sequential under temporary sleep/idle/lid
inhibition; user manages heat, no thermal guard/throttling introduced. Other stopped
experiments are not resumed. Data/events/large checkpoints remain local/ignored.

## Verification and measurements

660 tests pass per branch plus Ruff lint/format before expensive training. New
mechanism tests compare FP64 sequential values and ALL parameter/input gradients,
including write gate and both low-rank factors. Tests cover write suppression with
retained state, signed-query reversal of the same stored feature, explicit UV vs
factored readout values/gradients, exact counts, unchanged initial tensors/RNG,
causality/future gradients, gradcheck, full/chunk/single-token inference, snapshots,
cache reset/constant size, residual once, zero/2049-token behavior and BF16 with
FP32 states/finite gradients. Full-size optimizer benchmarks precede full training.
Existing baseline tests remain passing; nonfinite-gradient guard stays active.

Initial/final held-out-batch probes measure mixer/raw-state RMS, query distribution
and negative fraction, retention statistics, input-gate statistics/token variation
and gated-write amplitude, low-rank factor norms/singular values/effective rank,
and finite per-parameter gradients. Singular values use the small QR-factor product,
equivalent to nonzero singular values of U V, avoiding a full dense matrix SVD.
Effective rank is exp(entropy of normalized singular values), not semantic evidence.
Generic six-slice norm statistics, where present, are arbitrary feature groups,
not actual heads. Full curves and fixed-prompt greedy generations are preserved;
primary loss/PPL uses final-budget state, generations use best saved checkpoint.

## Interpretation scope

Independent writes test separating write magnitude from retention, with 887040 extra
parameters as a capacity confound versus No-K. Its exact count match to original
FluxV is useful, but write activation/state mechanisms differ. Signed readout is
the cleanest parameter-matched ablation: only query activation differs. Low rank
tests a learned mixing/compression readout together; a rank 32 direct map is not
merely adding interactions to the old identity map. State sizes remain matched.
Differences in gate values, signs or singular values do not prove semantic feature
specialization, reasoning, retrieval or long-context preservation. Single seed,
limited TinyStories exposure: record observations separately from hypotheses.
