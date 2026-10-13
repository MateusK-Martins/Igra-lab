# Ember: feature-wise diagonal decay

Ember is the cumulative model on `model/001-spoony-ember`. The pre-decay snapshot
was pushed at `ced912d`, before restoring the earlier trainable internal RMSNorms
and adding decay. Historical experiment names and manifests remain unchanged.

Q and V stay sigmoid-gated; K stays tanh. With zero initial states, the mechanism
is now:

```text
K_t = a_K * K_(t-1) + tanh(W_K x_t)
V_t = a_V * V_(t-1) + sigmoid(W_V x_t) * RMSNorm(K_t)
y_t = x_t + sigmoid(W_Q x_t) * RMSNorm(V_t)
```

Products and decays are feature-wise. Every enabled coefficient is sigmoid of
an independent learned [384] parameter vector, initialized to 0.99. Each layer
owns its own vectors; K and V vectors are independent. This is a diagonal,
time-invariant state recurrence, not a full SSM discretization or a selective
input-dependent gate. Input contributions are not multiplied by (1-a).

Run 1 enables V decay and fixes K decay to one. Run 2 enables both K and V decay.
Both start from scratch with the same seed-42 initialization of shared weights,
fixed tokenizer/data artifacts, 9,000 updates, 18.43M target tokens, batch 8,
context 256, BF16, optimizer and schedule as the earlier experiments. AdamW's
existing weight decay applies to the new decay logits too. The added parameters
are 2,304 for V only and 4,608 for K+V across six layers. RMSNorm's learned gains
are restored in both runs. Soft normalization remains available but is not used.

Training uses a differentiable parallel doubling scan: offsets 1,2,4,... process
all token positions in parallel at each stage. It has O(log T) sequential depth
and O(B*T*D*log T) work in this eager implementation, rather than a token loop.
There is no division by decay or inverse-power cumulative sum. No compile or
custom kernel is required. Cached inference continues the same recurrence by
adding a^t times the previous raw state; only two [B,1,D] states are stored.

550 tests passed, including independent sequential recurrence/gradient checks,
per-feature decay and initial-state gradients, gradcheck, near-zero/unit decay,
long scans, future-token isolation, chunked/single-token snapshot continuation,
independent coefficient vectors, and CUDA FP16/BF16 checks. Ruff passed.

Run 1 outputs: results/pretraining/ember-v-decay-v4/.
Run 2 outputs: results/pretraining/ember-kv-decay-v5/.
Each has its own benchmark, frozen recipe commit, manifest, checkpoints and report.
A single seed and recipe do not establish a general architecture ranking.

## V-only result

Recipe commit `db2111f`. Completed 9,000 updates and 18,429,440 target tokens
in 322.18 seconds including validation and saving. Final validation loss
2.519256, perplexity 12.419357. Final checkpoint is best
among saved checkpoints, loaded successfully for generation and checked against
the manifest. Learned per-layer feature-decay ranges are saved in the report.

## K+V recipe

The second configuration enables key_decay=True as well as value_decay=True,
with independent [384] learned logits for each per layer, still initialized to
0.99. RMSNorm, shared initialization draws, optimizer, schedule and data order
are unchanged. All 550 tests and Ruff passed with this configured model too.
This is a fresh run, not continuation or fine-tuning of the V-only weights.

## Completed comparison — 2026-10-13

K+V recipe commit `ec1e927`. All 550 tests and Ruff passed for both configured
models. Both fresh runs consumed exactly 18,429,440 target tokens in 9,000
updates. Their tokenizer/split hashes and training settings match the no-decay
sigmoid/RMSNorm baseline exactly. Final checkpoints are best among nine saved
checkpoints per run, were loaded for fixed greedy generations, and matched
their manifests. The final run's current source hashes also match its manifest.

| RMSNorm + sigmoid Q/V variant | Parameters | Final validation loss | Perplexity | Time |
| --- | ---: | ---: | ---: | ---: |
| No decay | 12,887,040 | 2.636655 | 13.966409 | 5m 24s |
| V decay | 12,889,344 | 2.519256 | 12.419357 | 5m 22s |
| K+V decay | 12,891,648 | 2.509973 | 12.304594 | 6m 24s |

V-only decay improved loss by 0.117399 nats in this seed; adding K decay gave a
further 0.009284-nat improvement but took more time. This small additional gain
is not evidence of a reliable advantage without repeated seeds. Both decay
variants remain worse in validation loss than Lantern's attention baseline
(2.018320). Generations remain repetitive and sometimes inconsistent.

All six learned V vectors, and all twelve learned K/V vectors in the second
run, are saved in the respective checkpoints. Their per-feature ranges and
standard deviations are recorded in the training reports, demonstrating
feature-specific updates rather than a single shared decay. Current model
configuration keeps both decays enabled, initialized to 0.99 for fresh runs,
and uses the restored trainable RMSNorms. There are no token-by-token Python
loops in the runtime scans; sequential references live only in tests.

Reports:
[V-only](results/pretraining/ember-v-decay-v4/training-report.json),
[K+V](results/pretraining/ember-kv-decay-v5/training-report.json),
[comparison](results/pretraining/ember-kv-decay-v5/decay-comparison.json).
Timing includes validation and checkpoint saves, excluding preparation, tests,
benchmarking and subsequent generation. Checkpoints and data remain local and
ignored; code, manifests, reports and comparison are pushed on the Ember branch.
For historical checkpoint loading, use the recorded recipe commit and matching
configuration; result-only commits are different Git identities under the
existing strict checkpoint compatibility policy.
