# Token-conditioned decay — codename pending

This model branches from pushed Ember commit c856b10 on codex/token-decay.
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
