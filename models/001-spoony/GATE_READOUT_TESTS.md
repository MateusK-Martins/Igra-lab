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
LR3e-4,betas(.9,.95),wd.01,warmup300,cosine floor.1,clip 1,accumulation1,
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

## Completed controlled comparison

All three fresh runs reached 9,000 updates and 18,429,440 target tokens with matched
data/settings/source fingerprints and unchanged original parameter initialization.
No trained warm start or post-result architecture tuning. All 18 validation points
and fixed-prompt generations are preserved.

| Model | Parameters | Final loss | Perplexity | Measured min | Tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FluxV | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 46,589 | 1.044 |
| E recurrent matrix memory | 14,211,456 | 2.039836 | 7.6893 | 35.04 | 7,792 | 1.044 |
| Lantern attention | 12,440,064 | 2.018320 | 7.5257 | 7.19 | 42,727 | 0.748 |
| No-K signed writes | 12,887,040 | 2.123176 | 8.3576 | 6.84 | 44,938 | 0.940 |
| squared-retention | 12,887,040 | 2.112602 | 8.2697 | 10.62 | 28,926 | 1.285 |
| write-gate | 13,774,080 | 2.090822 | 8.0916 | 6.28 | 48,876 | 0.999 |
| signed-readout | 12,887,040 | 2.165061 | 8.7151 | 5.99 | 51,243 | 0.940 |
| lowrank-readout | 13,034,496 | 2.175022 | 8.8024 | 6.05 | 50,733 | 0.951 |

E timing covers its resumed 8000-update phase, not full 9000 updates. All primary
losses use final-budget state. Other reference timings are full historical runs.

### What each test reveals

Positive loss differences below are worse than RMSNorm No-K.

- write-gate: -0.032354 nats/token versus No-K.

- signed-readout: +0.041885 nats/token versus No-K.

- lowrank-readout: +0.051846 nats/token versus No-K.

**Independent input gate:** Loss changes by -0.032354 against No-K.
It tests a separate magnitude controller for signed content writes; original tanh
writes already have variable magnitude, but the new projection independently
modulates that content. Functional tests verify that old state can be retained
while the write gate is closed. Learned gate statistics below show adaptation,
not proof of useful long-context preservation or a semantic write policy.
The additional 887,040 parameters confound a pure gating explanation; its full
13,774,080 count exactly matches original FluxV, with a different one-state mechanism.

**Signed query:** Loss changes by +0.041885 with identical
parameter count and state/recurrence. This is the cleanest capacity-matched test.
Tanh allows current input to reverse a stored feature's sign and removes sigmoid's
positive offset; it also changes saturation/derivatives and initialization scale.
Therefore any gain/loss cannot be attributed solely to sign availability. Original
S is already signed, so positive Q never implied only positive residual outputs.

**Low-rank mixing:** Loss changes by +0.051846 with 147,456
extra weights. Rank 32 introduces cross-feature linear interactions while replacing
the identity state readout with a bottleneck. Results therefore apply to that direct
rank 32 formula/initialization, not to all state mixing or identity-plus-low-rank
readouts. The recurrent state remains 384 features; no matrix memory or token-cache
growth is introduced. Effective rank below is descriptive utilization, not semantic
feature specialization or evidence of retrieval ability.

write-gate has the lowest final loss of these three in this seed.
It improves the completed No-K reference here; repeat seeds are needed before treating the change as robust.

The comparison to squared-retention normalization is historical, not a combined
experiment: none of these new mechanisms is combined with squared-retention here.
Interactions between normalization, signed query, input gating and low rank remain
untested. Same exposure is not equal compute, and prior session timing differences
limit precise speed attribution. Single seed: no general architecture-superiority,
reasoning or retrieval claim.

### Stability and learned mechanism diagnostics

All initial/final mixer outputs and parameter gradients are finite. The existing
nonfinite-gradient guard and clip 1 remain active at every BF16 optimizer step.
RMS below averages six layers of a fixed held-out batch; raw state samples the
last cached token. Gradient norms are un-clipped diagnostic norms.

| Model | Mixer RMS initial → final | Final raw state RMS | Gradient norm initial → final | Negative Q fraction | Final retention mean | Retention token variation std |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| write-gate | 0.5169 → 0.4952 | 4.3691 | 3.1589 → 1.7115 | 0.0000 | 0.90359 | 0.13454 |
| signed-readout | 0.4595 → 0.6651 | 7.2642 | 1.5524 → 2.3553 | 0.4944 | 0.89625 | 0.14301 |
| lowrank-readout | 0.5073 → 0.5403 | 12.9834 | 3.3672 → 1.7693 | 0.0000 | 0.90254 | 0.13544 |

Independent-write gate per-layer measurements (initial gate exactly 0.5):

| Layer | Input gate mean/min/max | Input gate token variation std | Gated-write RMS |
| --- | --- | ---: | ---: |
| body.1.branch | 0.39511/0.01238/0.99112 | 0.23360 | 0.26857 |
| body.5.branch | 0.44793/0.00613/0.99703 | 0.18023 | 0.31946 |
| body.9.branch | 0.43631/0.00401/0.99740 | 0.16654 | 0.30860 |
| body.13.branch | 0.45005/0.01380/0.99686 | 0.17203 | 0.30970 |
| body.17.branch | 0.46676/0.01099/0.99123 | 0.17035 | 0.31056 |
| body.21.branch | 0.48760/0.01274/0.99884 | 0.14937 | 0.33781 |

Low-rank readout per-layer measurements:

| Layer | V factor norm | U factor norm | Effective rank initial → final (max 32) |
| --- | ---: | ---: | --- |
| body.1.branch | 6.21275 | 19.56211 | 31.231 → 31.458 |
| body.5.branch | 5.56201 | 19.06492 | 31.213 → 31.290 |
| body.9.branch | 5.74546 | 19.00222 | 31.291 → 31.341 |
| body.13.branch | 5.69309 | 19.13486 | 31.329 → 31.303 |
| body.17.branch | 5.61718 | 18.86038 | 31.297 → 31.356 |
| body.21.branch | 5.43566 | 18.67505 | 31.322 → 31.401 |

All 32 nonzero singular values per sampled layer are retained. QR-factor SVD is
mathematically equivalent to nonzero singular values of U V. Entropy effective rank
uses normalized singular values, not squared singular values. A large effective
rank does not show that every direction is useful or semantically distinct.
Gate/sign activity similarly demonstrates use of the available mechanism, not
its causal contribution to quality or long-term memory. Full per-parameter gradient,
state, gate, norm and query distributions remain in JSON. All three inference caches
remain 1.5 KiB/sample/layer, identical to original No-K.

### Fixed-prompt greedy generations

Inherited best-saved-checkpoint procedure, 64 new tokens max. Primary loss/PPL uses
final-budget state. Samples are illustrative stories, not controlled QA evaluations.

#### write-gate — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running towards her. Lily was scared and ran away.
> But then, the dog saw a big dog. The dog was barking and running towards the dog. Lily was scared and ran away.

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and wanted to play with Tom.
> Tom and the dog played together all day. They ran and ran, and laughed together. Tom was not sad anymore. He was not lonely anymore. He was happy to have a new friend.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy, but he could not find it. He looked everywhere, but he could not find it.
> Then, a big dog came to the park. The dog saw the toy and wanted to play with it. The dog said, "I want to play with you

#### signed-readout — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball. She wanted to play with it, but it was too high. She tried to catch it, but it was too high.
> Lily's mom saw her crying and asked, "What's wrong, Lily?"
> "

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and he wanted to play with the ball. He ran to the dog and said, "Hi, dog! What are you doing?" The dog looked at Tim and said, "I am not scared. I am not scared. I am a bad dog. I am not a good dog. I am a

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his toy car. But he did not know how to fix it. He thought of a way to make it go away.
> He went to his mom and asked, "Mom, can I have the toy car? It's too big and heavy for me."
> His

#### lowrank-readout — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running towards her. Lily was scared and ran away.
> Lily's mommy came running and asked, "What's wrong, Lily?" she asked. "I lost my toy," Lily said.

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not happy. He wanted to play with the dog. He ran to the dog and said, "Hi, dog. Do you want to play with me?"
> The dog wagged his tail and wagged his tail. He wagged his tail and wagged his tail. He wagged his tail and wagged his tail. He liked the

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his ball. He looked everywhere for it, but he could not find it.
> Then, he saw a big dog. The dog was not nice. The dog wanted to play with the ball. He ran to the dog and said, "Hi, dog! Can I
