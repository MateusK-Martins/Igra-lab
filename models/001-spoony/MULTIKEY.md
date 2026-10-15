# FluxV four-key-bank experiments

Reference: original V-only FluxV recipe at
29890bc8a7327667102eceb823317a009b96d32e. Both branches start from scratch,
never from trained weights. The original model and earlier experiments are preserved.

## Architecture

Four independent full-width K projections create four 384-feature recurrent keys.
Each has learned feature-wise diagonal retention d_h=sigmoid(decay_logits_h),
initialized to 0.99, constant across tokens but trainable. One vectorized parallel
scan computes all 1536 features; no token/head processing loops or scan changes.

K_h[t] = d_h * K_h[t-1] + tanh(W_K,h x[t])
s[t] = sum_h(w_h * K_h[t])
k_used[t] = global_RMSNorm(q_key[t] * s[t])
V[t] = sigmoid(W_D x[t] + b_D) * V[t-1] + sigmoid(W_V x[t]) * k_used[t]
branch[t] = sigmoid(W_Q x[t]) * global_RMSNorm(V[t])
output[t] = x[t] + branch[t]

Products are element-wise. Each w_h is a learned 384-feature signed output vector,
initialized Normal(0, 1/sqrt(4)); its variance keeps the summed initial scale
reasonable. There is no head normalization, softmax, router or output projection.
Q gates the sum before the single trainable global K RMSNorm. The residual wrapper
adds x exactly once. V decay remains token-dependent, with zero gate weights and
bias logit(0.99). Both internal learned RMSNorm gains/epsilon are preserved.

- Tied Q: q_key = sigmoid(W_Q x), the same query tensor also reads normalized V.
- Independent Q: q_key = sigmoid(W_QK x), a fresh bias-free 384→384 projection;
  original sigmoid(W_Q x) continues reading V.

Four heads were selected before training as a modest bank of memories. Ten was
only the user's example. The full stream width stays 384; these are four complete
K vectors, not a partition into narrow heads. No head-count sweep is performed.

| Model | Parameters | Change from original |
| --- | ---: | ---: |
| Original V-only FluxV | 13,774,080 | — |
| Four keys, tied Q | 16,446,720 | +2,672,640 (+19.40%) |
| Four keys, independent Q | 17,331,456 | +3,557,376 (+25.83%) |

Per layer, extra K projection weights are 3×384²=442,368. Four decay vectors plus
four output vectors add 3,072. Independent Q adds 147,456 more. No FFN or outer
block changes compensate for this capacity; these are explicitly not parameter-
matched 14M ablations. The reference label describes the starting recipe.

The cache contains K:[B,1,1536] and V:[B,1,384], FP32 under BF16. That is 1920
state elements per sample/layer (7.5 KiB FP32), versus 768 (3 KiB) originally,
a 2.5× increase, constant with sequence length. StateEntry supports independent
key/value feature widths with validation of both. Separate module cache IDs and
state-dictionary shapes reject original vector checkpoints; there is no migration
or reinterpretation of previous trained states.

## Controlled initialization and protocol

Seed 42. Original Q/K/V are drawn first. Additional projection/weight creation
uses a CPU RNG fork, preserving original Q/V, V gate, norms, FFNs and subsequent
layer initialization. The first K bank starts exactly at the original K weights;
other banks have independent fresh weights. Both modes construct the fresh reader
inside the fork, retaining it only for independent Q. Every common tensor starts
identically between the two models. The data loader uses its independent generator.
The audit explicitly compares 68 unchanged tensors to the original initialization.

Same pinned TinyStories 30,000 train / 256 validation documents, tokenizer 8192,
verified exact prepared-artifact hashes, context 256, batch 8, BF16, AdamW LR3e-4,
betas(.9,.95), wd.01, warmup300, cosine floor.1, clip1, accumulation1. Each fresh
run targets 9,000 updates and 18,429,440 actual target tokens, including the same
partial epoch-ending batches. Validation every500; checkpoints every1000. The
training, scan, optimizer, validation and generation infrastructure are reused.
No QA post-training stage or thermal throttling is added. Sleep/idle/lid suspend
is inhibited only while the GPU work runs, as requested; the user manages heat.

## Verification and reporting

623 tests pass per branch, including sequential FP64 value and parameter/input
gradient comparisons, distinct per-head decay and writes, Q tying, causal/future
isolation, gradcheck, cached chunks/single-token inference/snapshots/reset,
constant cache size, residual-once behavior, legacy-weight rejection, BF16 with
FP32 cached states, long/zero-input behavior, configuration and exact counts.
Ruff lint/format checks pass. Preflight checks exact data/settings fingerprints.

Initial and final fixed-held-out-batch probes report mixer RMS, finite parameter
gradients, key-state/weighted-key RMS per bank, output-weight norms, learned diagonal
K retention, and token-dependent V retention. These amplitudes do not establish
semantic specialization. The full validation curve and original fixed-prompt greedy
generations are preserved; primary metrics use the final-budget checkpoint, while
generations follow the existing best-saved-checkpoint procedure.

Commands, from each model directory with its .venv:

```bash
python -m pytest -q
ruff check src configs scripts tests
ruff format --check src configs scripts tests
python -m scripts.probe_variants initial
python -m scripts.benchmark
python -m scripts.run_baseline
python -m scripts.probe_variants final
```

Benchmark/probe processes discard their weights. Fresh training begins separately
from seed42. Frozen manifests identify exact recipe commits and source hashes.
Datasets, tensorboard events and large checkpoints stay local and ignored.
Results are single-seed observations: extra parameters/state/compute prevent a
capacity-controlled comparison to the historical original. Tied vs independent
Q also differs by 884,736 parameters. Lower loss is not proof of retrieval or reasoning.

## Completed comparison

Both fresh runs completed 9,000 updates / 18,429,440 target tokens. Frozen source
hashes and matching data/settings were verified; final gradient probes are finite.

| Model | Parameters | Loss | Perplexity | Minutes | Tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 46,589 | 1.044 |
| tied | 16,446,720 | 2.169084 | 8.7503 | 15.84 | 19,396 | 1.777 |
| independent | 17,331,456 | 2.135759 | 8.4635 | 15.78 | 19,462 | 1.811 |

Independent Q lowered loss by 0.033324 versus tied Q in this seed. Tied Q barely
improved the original. This supports testing separate key/value queries further,
but independent Q also adds 884,736 parameters. Both add K projections, K decay
and key-query gating relative to the original; this is not a parameter-matched
isolation of any one change. Single-seed results do not establish superiority.
K retention means remained similar across banks (~0.989); amplitudes alone do not
establish distinct memory timescales or semantics. Full curves, generations and
initial/final diagnostics are preserved in comparison.json and per-run reports.
