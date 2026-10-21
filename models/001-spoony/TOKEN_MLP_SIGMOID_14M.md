# Two 14M Lantern sigmoid-gate TokenMLP comparisons

Branch: `codex/lantern-tokenmlp-sigmoid-14m`, based on the all-shared TokenMLP
14M Lantern experiment. The mechanism replaces the tanh outer-product matrix
with two sigmoid gates multiplied elementwise into the retrieved 64-feature
representation from the shared down projection. The shared up projection maps
the gated representation back to width 384; the declared residual adds it to
the layer input. Tables and down/up projections remain shared across all six
TokenMLP blocks, preserving the previous TokenMLP sharing pattern.

The matched tests differ only in how the second gate is obtained:

1. **Learned pair:** `z = W_down(x)`;
   `y = W_up(z * sigmoid(A[token]) * sigmoid(B[token]))`.
   Both width-64 vectors come from learned token tables.
2. **Context gate:** `z = W_down(x)`;
   `y = W_up(z * sigmoid(A[token]) * sigmoid(z))`.
   The first vector comes from a learned token table. The second is generated
   from the current causal hidden state through the shared down projection.

Both retain width 384, six layers, GQA 8/2 with head width 48, window 256,
RoPE 10000, tied 8192-token embedding/head, context 256, the same TinyStories
subset/tokenizer, seed 42, batch 8, BF16, AdamW settings and 9,000-update
schedule as the prior all-shared TokenMLP run. Each run gets a fresh output
directory. No benchmark weights are used for training.

| Variant | Learned token tables | Parameters at V=8192 | Output directory |
|---|---:|---:|---|
| Learned pair | 2 shared tables | 6,459,904 | `results/pretraining/lantern-tokenmlp-sigmoid-learned14m-v1/` |
| Context gate | 1 shared table | 5,935,616 | `results/pretraining/lantern-tokenmlp-sigmoid-context14m-v1/` |

The context gate reuses `W_down(x)` as both the gated value and context input;
it adds no new context projection. Run reports will record artifact hashes,
recipe settings, validation curves, best checkpoints, samples and full run time.
These are single-seed preliminary results and should be compared with the
original Lantern and previous all-shared TokenMLP rows in `TOKEN_MLP_ALL_SHARED.md`.

## Completed matched runs

Both variants completed 9,000 updates and consumed the same 18,429,440 target
tokens from the same pinned TinyStories artifacts, with seed 42 and the same
schedule. The best checkpoint was the final checkpoint for each. The context
gate is 524,288 parameters smaller and has 0.00580 lower validation loss in
this single-seed comparison; this small difference needs additional seeds.

| Model | Parameters | Validation loss | Perplexity | Updates | Target tokens | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Original Lantern | 12,440,064 | 2.018320 | 7.525675 | 9,000 | 18,429,440 | 431.33 s |
| Shared tanh-matrix TokenMLP | 6,459,904 | 2.242750 | 9.419195 | 9,000 | 18,429,440 | 308.00 s |
| Sigmoid learned pair | 6,459,904 | 2.285694 | 9.832505 | 9,000 | 18,429,440 | 287.97 s |
| Sigmoid context gate | 5,935,616 | 2.279891 | 9.775617 | 9,000 | 18,429,440 | 285.28 s |

Both sigmoid runs finished with the final step also best among saved checkpoints.
They are faster and smaller than the Lantern run, but have higher validation
loss. Greedy samples are saved with each training report and exhibit repetition;
they are qualitative probes only. Peak allocated memory was 605,611,008 bytes
for the learned pair and 597,353,472 bytes for the context gate.

Reports, manifests, progress logs and benchmarks are in
`results/pretraining/lantern-tokenmlp-sigmoid-learned14m-v1/` and
`results/pretraining/lantern-tokenmlp-sigmoid-context14m-v1/`.
