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
