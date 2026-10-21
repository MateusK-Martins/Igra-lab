# Seven-layer shared TokenMLP — Lantern 14M test

Branch: `codex/lantern-tokenmlp-extra-layer-14m`.

This test keeps the baseline TokenMLP block unchanged and increases the stack
from six to seven layers. Width (384), shared 64-dimensional token tables and
projections, attention, normalization, tokenizer, TinyStories subset, seed,
optimizer, and 9,000-update schedule remain the same. At vocabulary 8,192,
the additional attention layer and normalization add 369,408 parameters,
bringing the model to 6,829,312 unique parameters.

The matched six-layer TokenMLP baseline has 6,459,904 parameters and reached
validation loss 2.242750 (perplexity 9.419195) after 9,000 updates. This branch
records whether one more standard TokenMLP layer helps under the same token
budget; it is a single-seed exploratory comparison.

The training run and benchmark are recorded in
`results/pretraining/lantern-tokenmlp-extra-layer-14m-v1/`.

## Completed run

The matched 9,000-update run completed in **341.26 seconds**, consuming
18,429,440 target tokens. Best saved validation loss was **2.225760**
(perplexity **9.260513**) at update 9,000; peak allocated VRAM was 658,606,080
bytes. Compared with six-layer shared-64 TokenMLP (6,459,904 parameters,
validation loss 2.242750, 308 seconds), the seventh layer added 369,408
parameters and 33.26 seconds, and improved validation loss by 0.01699 in this
single seed. Greedy samples and the validation curve are in the training report.
