# TokenMLP with an RMSNorm context vector — 14M Lantern

Branch: `codex/lantern-tokenmlp-rmsnorm-14m`

This repeats the shared-64 TokenMLP Lantern 14M experiment, replacing the
second learned token-vector table with a context-derived vector. For hidden
state `x`, the shared down projection produces `z = W_down(x)` with width 64.
The token ID retrieves `a = A[token]`; the other vector is
`b = 0.02 * RMSNorm(z)`. Each TokenMLP computes
`tanh(a bᵀ) @ z`, then applies the shared up projection and the existing
residual. The RMSNorm has one shared learnable scale, epsilon `1e-6`, and is
shared across all six TokenMLP blocks. Attention, model dimensions, tokenizer,
data, optimizer, schedule, and seed match the baseline experiment.

The run used the same 9,000 updates and 18,429,440 training tokens as the
original shared-64 baseline. It completed in 309.6 seconds on an RTX 4050 Laptop
GPU. Results:

| Run | Parameters | Updates / tokens | Time | Best validation loss | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared-64 baseline (two learned token vectors) | 6,459,904 | 9,000 / 18,429,440 | 308.0 s | 2.242750 | 9.4192 |
| RMSNorm context vector (this branch) | 5,935,680 | 9,000 / 18,429,440 | 309.6 s | 2.260821 | 9.5910 |

This variant is 524,224 parameters smaller. Validation loss is 0.01807 higher
(about 0.8%) under this single-seed matched-token comparison; training time is
effectively unchanged. This is one run per architecture, so the small metric
difference should not be read as a multi-seed significance result.

The run report includes the validation curve, benchmark, data subset hashes,
and greedy samples from the best checkpoint. The checkpoint is
`results/pretraining/lantern-tokenmlp-rmsnorm-hidden14m-v1/checkpoints/step-00009000.pt`.
