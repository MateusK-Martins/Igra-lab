# Flux: token-conditioned decay

Flux branches from pushed Ember commit c856b10 on model/001-spoony-flux.
Ember remains unchanged on model/001-spoony-ember. The model-local directory is
still models/001-spoony, with independent code, configs and copied data artifacts.

Q/V stay sigmoid-gated, K stays tanh, and both internal norms are trainable
RMSNorm. The diagonal time-invariant retention coefficient is replaced by:

```text
D_K(x_t) = sigmoid(W_DK x_t + b_DK)
D_V(x_t) = sigmoid(W_DV x_t + b_DV)
K_t = D_K(x_t) * K_(t-1) + tanh(W_K x_t)
V_t = D_V(x_t) * V_(t-1) + sigmoid(W_V x_t) * RMSNorm(K_t)
y_t = x_t + sigmoid(W_Q x_t) * RMSNorm(V_t)
```

Products are element-wise. Each enabled gate maps width 384 to 384, producing
[B,T,D] retention coefficients depending only on that token's current block
input. Each layer owns independent projections; K and V do not share gates.
Gate weights start at zero and biases at logit(0.99), matching Ember's initial
retention while allowing input-dependent gates to develop during training.
This initialization consumes no extra RNG draws for the common model weights.
Gate projection can use autocast, but bias addition and sigmoid run in FP32
(or FP64 for double reference checks) to preserve the initial coefficient.

The affine scan composes left then right pairs using
(A_right*A_left, B_right + A_right*B_left). Parallel doubling computes inclusive
prefixes in O(log T) sequential depth and O(B*T*D*log T) eager work. It uses no
Python loop over tokens, inverse coefficients or inverse-power cumulative sums.
Cached execution applies prefix products to the previous state; memory remains
two [B,1,D] raw states per layer. Reset behavior and context remain unchanged.

Run 1: token-dependent V decay only, K is an ordinary cumulative sum.
Run 2: token-dependent decay on both K and V, with independent gates.
Each starts fresh with seed 42, the exact same tokenizer/split artifacts, batch
order, context 256, BF16, optimizer and 9,000-update schedule as Ember. No weights
are initialized from a trained checkpoint. The older constant diagonal option
remains available through decay_mode="diagonal" for historical tests and models.
The active model selects decay_mode="token".

V-only adds 887,040 gate parameters to the 12,887,040 no-decay model, for
13,774,080 total. K+V adds 1,774,080, for 14,661,120 total. These are equal-token,
not parameter-matched, comparisons with Ember's 12.89M models. Gate weights and
biases receive the existing optimizer's weight decay, like other parameters.

603 tests passed and Ruff lint/format checks passed. New tests cover time-varying
coefficients, boundary zero/unit retention, nonzero initial states, all scan
input/coefficient/state gradients, gradcheck, constant-gate agreement, input
conditioning, gate initialization/RNG, cached chunks and snapshots with nonzero
gate weights, future isolation, and CUDA FP16/BF16 precision/gradients.

Outputs:
- results/pretraining/token-v-decay-v6/
- results/pretraining/token-kv-decay-v7/

Each run records its own recipe commit, manifest, benchmark, checkpoints and
fixed-prompt generations. Data and weights stay local and ignored; small reports
are versioned. A single seed and recipe do not establish an architecture ranking.

## Token-dependent V-only completed

Recipe commit `29890bc`. Fresh 9,000 updates / 18,429,440 target tokens finished
in 395.58 seconds including validation and saving. Final validation loss
2.173892, perplexity 8.792440. Final checkpoint is best
among nine saved, was loaded for generation and gate probing, and matched its
manifest. Ember's fixed V decay scored 2.519256 / 12.419357 at equal tokens.

A probe of the first ordered validation batch confirms token-dependent
retention: all six gate weight matrices changed from zero, and each feature's
retention varies across tokens. Mean within-feature token standard deviations
range about 0.085 to 0.246 across layers; layer-average retention ranges about
0.636 to 0.948. Full per-layer statistics and the probe input fingerprint are
in results/pretraining/token-v-decay-v6/gate-probe.json. This probe is outside
the measured training time.

## K+V recipe

The second configuration enables both token-conditioned gates. They are
independent 384-to-384 projections per layer, initialized to the same 0.99
retention, with RMSNorm and all training settings unchanged. All 603 tests and
Ruff passed with this configured model as well. This is a fresh run from seed
42, not fine-tuning or continuation of the V-only checkpoint.

## Completed Flux comparison — 2026-10-13

The branch was named Flux after both runs. Frozen reports retain the actual
training branch name codex/token-decay and original artifact directory names.
Recipe commits: V-only 29890bc, K+V 9354fab. All 603 tests and Ruff passed for
both configurations. Both fresh seed-42 runs consumed 18,429,440 target tokens
in 9,000 updates, with data/tokenizer hashes and training settings identical
to Ember. Final checkpoints are best among nine saved per run, loaded for
generation/probes and checked against their manifests. Current source hashes
match the K+V training manifest.

| Variant | Parameters | Validation loss | Perplexity | Time |
| --- | ---: | ---: | ---: | ---: |
| Ember fixed V | 12,889,344 | 2.519256 | 12.419357 | 5m 22s |
| Flux token V | 13,774,080 | 2.173892 | 8.792440 | 6m 36s |
| Ember fixed K+V | 12,891,648 | 2.509973 | 12.304594 | 6m 24s |
| Flux token K+V | 14,661,120 | 2.169193 | 8.751216 | 8m 48s |

Both token-conditioned variants score better in this seed, but are larger;
this does not isolate conditioning from added capacity. Adding K improves
Flux loss by only 0.004700 nats at higher time and parameter cost. V-only is a
strong cost/quality choice here; repeated seeds are needed for stronger claims.
Flux remains behind Lantern's attention baseline (loss 2.018320) under this
recipe. Generations remain repetitive and sometimes inconsistent.

Probes use the same first ordered validation batch in BF16 eval/no_grad with
no cache. All trained gate matrices are nonzero and all gates vary across
tokens within each feature. K+V layer-average K retention is about 0.933–0.980
with mean within-feature token standard deviation 0.023–0.072; V retention is
about 0.664–0.941 with variation 0.098–0.215. These confirm conditioning without
establishing a semantic interpretation. Sigmoid can round extreme values to
one in floating point; boundary zero/unit coefficients are covered by tests.

Reports and probes:
- [V report](results/pretraining/token-v-decay-v6/training-report.json)
- [V gate probe](results/pretraining/token-v-decay-v6/gate-probe.json)
- [K+V report](results/pretraining/token-kv-decay-v7/training-report.json)
- [K+V gate probe](results/pretraining/token-kv-decay-v7/gate-probe.json)
- [Comparison](results/pretraining/token-kv-decay-v7/token-decay-comparison.json)

Times include validation/saving, excluding preparation, benchmark, tests and
subsequent samples/probes. Code, manifests and reports are pushed on Flux; data
and checkpoints remain local and ignored in this worktree. Ember's original
ignored weights remain in its own worktree, not copied here. Current Flux
configuration retains both token-dependent gates and RMSNorm. For historical
checkpoint compatibility use the recorded recipe commit and configuration;
the existing strict resume policy includes the exact Git identity.
