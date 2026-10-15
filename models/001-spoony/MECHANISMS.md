# Controlled recurrent mechanism experiments

All branches originate at original V-only FluxV commit
29890bc8a7327667102eceb823317a009b96d32e. The preceding four-K comparison was
completed and pushed first, preserving its settings and checkpoints. It reached
loss/PPL 2.169084/8.7503 (tied Q), 2.135759/8.4635 (independent Q).

## Frozen protocol and parameter policy

The user explicitly chose to preserve the original FFN and document differences,
rather than reallocate parameters into different FFN widths. Width384, six layers,
SwiGLU1024, vocabulary8192, context256, batch8 remain unchanged. Seed42, BF16,
AdamW LR3e-4, betas(.9,.95), wd.01, warmup300, cosine floor.1, clip1, accumulation1,
9000 updates /18429440 actual target tokens. Partial epoch-ending batches explain
why exposure is slightly below the nominal18432000. Validation500/checkpoint1000.
Same pinned TinyStories30000 train/256 validation documents, exact tokenizer and
prepared-artifact fingerprints, sample ordering and independent loader RNG.
No QA post-training. Training, evaluation, generation, optimizer and scan code are
reused; no scan optimization, attention addition or environment redesign.

| Mechanism | Parameters | Change vs FluxV | FP32 state elements/sample/layer |
| --- | ---: | ---: | ---: |
| Original FluxV | 13,774,080 | — | 768 |
| Lantern historical attention baseline | 12,440,064 | -9.68% | Token-growing KV cache |
| No-K, token decay | 12,887,040 | -6.44% | 384 |
| Two K heads, independent Q | 15,552,768 | +12.91% | 1152 |
| Simple diagonal state | 12,002,304 | -12.86% | 384 |
| Signed-mass normalization | 13,769,472 | -0.033% | 1536 |

Differences are consequences of the mechanisms, not capacity padding. No-K removes
six K projections (884736 parameters) and six K RMSNorm gains (2304). Simple state
also replaces token-gate matrices with diagonal vectors, saving884736 more. Two K
heads add a full K projection, a separate Q projection, and two decay/output-weight
vectors per layer. Mass normalization removes both internal learned RMSNorm gains
(4608 total), with no new trainable projections. Outer blocks and residuals stay fixed.

## Exact mechanisms

All products/divisions below are feature-wise. q=sigmoid(W_Q x), initial states0,
learned diagonal d=sigmoid(logits) initially0.99. Token gate g=sigmoid(W_D x+b_D)
has zero weights and bias logit(.99), and becomes input-dependent during training.
Low-precision recurrence and token-gate arithmetic accumulate in FP32. Inference
caches contain a constant number of vector elements, independent of token count.

### No-K FluxV

```text
S[t] = g(x[t]) * S[t-1] + tanh(W_V x[t])
branch[t] = q[t] * RMSNorm(S[t])
output[t] = x[t] + branch[t]
```

There is no K projection, cumulative K state or K normalization. Signed tanh V
writes let memory cancel/overwrite contributions. Q, V, token-dependent V decay,
V RMSNorm, outer blocks and residual behavior are retained from FluxV.

### Two independently decaying K heads, independent query

```text
K_h[t] = d_h * K_h[t-1] + tanh(W_K,h x[t]), h in {0,1}
s[t] = sum_h(w_h * K_h[t])
q_key[t] = sigmoid(W_QK x[t])
k_used[t] = RMSNorm(q_key[t] * s[t])
V[t] = g(x[t]) * V[t-1] + sigmoid(W_V x[t]) * k_used[t]
branch[t] = q[t] * RMSNorm(V[t])
```

Each K head is a complete384-vector, not a width partition. It has its own learned
per-feature diagonal decay and signed output-weight vector. Output weights start
Normal(0,1/sqrt(2)), independent key projections use the usual bias-free Linear
initialization. All heads share one vectorized flattened diagonal scan. Query gates
the weighted sum before one global K RMSNorm. QK is independent of the original Q
that reads V. This is the same mechanism as the completed four-head independent-Q
run, changing only head count and resulting capacity/state/initialization shape.

### Simple diagonal-decay vector state

```text
S[t] = d * S[t-1] + tanh(W_V x[t])
branch[t] = q[t] * RMSNorm(S[t])
```

This is the minimal single-state control with learned fixed feature retention.
It has no K or cascaded recurrence. Compared with No-K above, only retention
changes from an input-dependent projected gate to a fixed learned diagonal. Q,
signed writes, RMSNorm, FFN and all outer blocks match, including initialization.
The parameter decrease reflects removal of gate matrices. This isolates decay
adaptation more cleanly than also changing readout or adding EMA write scaling.

### Attention-inspired signed-mass normalization

```text
k[t] = tanh(W_K x[t]); v[t] = sigmoid(W_V x[t])
K_num[t] = K_num[t-1] + k[t]
K_den[t] = K_den[t-1] + abs(k[t])
k_used[t] = K_num[t] / max(K_den[t], 1e-6)
V_num[t] = g(x[t]) * V_num[t-1] + v[t] * k_used[t]
V_den[t] = g(x[t]) * V_den[t-1] + abs(k_used[t])
branch[t] = q[t] * V_num[t] / max(V_den[t], 1e-6)
```

Both internal RMSNorms and their learned gains are absent. Positive absolute-write
mass avoids unstable division by a cancelling signed denominator. Each denominator
tracks historical write mass with exactly its numerator's decay. The key stage is
nondecaying, matching original FluxV; both V states share the original token gate.
The denominator floor is fixed, applied at readout only, never added to recurrent
state. There is no trainable output scaling. This construction bounds normalized
key/readout components in[-1,1] because V writes are sigmoid-bounded. It trades
RMS amplitude normalization for historical weighted averaging, so scale dynamics
and cancellation differ. It is linear-attention-inspired denominator accumulation,
not canonical q^T M/(q^T z): no outer-product matrix memory, no softmax, and Q
remains original element-wise gating after normalized state readout.

Key numerator/denominator are packed in a768-feature vector; V pair likewise.
No-K/simple store only one384-feature state: the generic key tensor is empty,
not an allocated dummy K. StateEntry supports independent widths, including zero
key width, while validating batch/sequence/value widths, dtype and counts. Distinct
module cache IDs and strict state-dictionary shapes prevent old checkpoint reuse.

## Correctness and diagnostics

638 tests passed per branch before expensive training, plus Ruff lint/format.
Mechanism tests compare FP64 values and all parameter/input gradients against
independent sequential references, including decay and writes. Tests cover head
combination/query tying, future-token gradient isolation, gradcheck, cached chunks,
single-token execution, snapshot restoration/reset, constant memory, residual once,
no-K absence, nonnegative denominators/bounded normalized writes, zero and2049-token
sequences, and BF16 with FP32 cached states. Existing baseline tests remain passing.
Full-size batch/context optimizer benchmarks precede training; finite-gradient
checks remain active at every training update. Numerical failure is reported, not
hidden by reducing context, tokens or changing a completed run's architecture.

Original Q/K/V/gate draws happen before deleting/replacing components, so unchanged
parameters and downstream FFNs retain original seed42 values. Extra K/query/output
initialization uses a CPU RNG fork. Preflight independently verifies unchanged
tensors against original initialization and compares exact artifact/settings
fingerprints against both historical FluxV and Lantern. Reference Lantern recipe
2eb1a7badd44a58a59841a5e817effb218fc071e used matching data/settings and9000 updates,
loss2.018320/PPL7.525675; its historical result is preserved.

Initial/final held-out-batch probes record mixer RMS and finite gradient norms,
learned retention and state statistics. Two-K additionally records per-bank raw and
weighted K RMS, decay and output-weight norms. Mass-normalized probes report
positive denominator/numerator distributions and floor usage; single-state probes
report recurrent norms. Generic six-slice norm summaries, where present, are merely
feature groups and must not be interpreted as physical heads or semantic roles.
Full curves and fixed-prompt greedy generations follow the inherited procedure;
primary metrics use final-budget state, generations use best saved checkpoint.

Commands from each model directory with its .venv:

```bash
python -m pytest -q
ruff check src configs scripts tests
ruff format --check src configs scripts tests
python -m scripts.probe_variants initial
python -m scripts.benchmark
python -m scripts.run_baseline
python -m scripts.probe_variants final
```

Benchmarks/probes run in separate processes and discard weights. Training starts
fresh from seed42. Frozen manifests identify recipes, source hashes and settings.
Checkpoints/datasets/events stay local and ignored. Temporary sleep/idle/lid
inhibition covers GPU work; temperature management remains with the user.

## Interpretation plan

No-K vs original tests whether cascaded K preprocessing is useful, but also changes
V write activation from sigmoid times signed normalized K to tanh and reduces
capacity. No-K vs simple diagonal state tests input-dependent retention, with gate
capacity as a confound. Two-K vs four-K tests memory-bank count/capacity; vs original
also changes K decay, query gating and capacity. Mass vs original primarily tests
normalization and gain removal, but signed-mass averaging imposes a different
amplitude/cancellation bias. Lantern provides a matched-data attention reference,
not an equal-parameter or equal-compute model. Single seed, limited TinyStories
budget: loss is not reasoning/retrieval evidence or proof of architectural superiority.
