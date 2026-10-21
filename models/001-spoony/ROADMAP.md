Active experiment: [Lantern 70M with shared TokenMLP, equal exposure](TOKEN_MLP_70M.md).

# Lantern Wikipedia experiment

Current recipe: 70,006,272 parameters, width 768, nine layers, FFN width 2432,
2048-token context and unbounded causal attention. The shared corpus contains
600M training tokens plus 2M held-out validation tokens. No QA mixing.

The default schedule is one complete pass over all full windows: 36,621
optimizer updates. See
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) for the verified execution path.
Only short benchmark/verification updates have run; no full training run has
been launched. Assess held-out loss and generated text after training before
choosing a separate instruction-tuning stage.
