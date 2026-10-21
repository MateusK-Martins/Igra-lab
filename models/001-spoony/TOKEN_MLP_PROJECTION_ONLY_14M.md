# Projection-only TokenMLP — Lantern 14M ablation

Branch: `codex/lantern-14m-tokenmlp-projection-only`.

This ablation retains the six-layer shared-64 TokenMLP architecture but removes
the two token embedding tables and the token-dependent `tanh(a ⊗ b)` matrix.
Each block still applies the shared encoder and decoder projections:

```text
z = W_down x        # 384 -> 64
output = x + W_up z # 64 -> 384, with the declared residual
```

Attention, normalization, input embedding/head tying, TinyStories subset,
tokenizer, seed, optimizer, and 9,000-update schedule match the TokenMLP
baseline. At vocabulary 8,192, this model has 5,411,328 parameters; the baseline
with the matrix has 6,459,904. The comparison isolates the contribution of the
token-specific matrix and tables while retaining the bottleneck projections.

The run is recorded in
`results/pretraining/lantern-tokenmlp-projection-only-14m-v1/`.

## Completed run

The matched 9,000-update run completed in **275.10 seconds**, consuming
18,429,440 target tokens. Best saved validation loss was **2.269199**
(perplexity **9.671652**) at update 9,000; peak allocated VRAM was 587,392,000
bytes. The six-layer standard shared-64 TokenMLP baseline reached validation
loss 2.242750 (perplexity 9.419195) in 308 seconds. On this one seed, removing
the matrix and tables reduced parameters by 16.2% and run time by 10.7%, with
validation loss higher by 0.02645. This is a single-seed TinyStories subset
comparison, not a broad quality estimate. Greedy samples and the validation
curve are in the training report.
