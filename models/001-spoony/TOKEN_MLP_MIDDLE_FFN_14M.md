# One full-width FFN among shared TokenMLP blocks — Lantern 14M test

Branch: `codex/lantern-tokenmlp-middle-ffn-14m`.

This control starts from the original shared-64 TokenMLP baseline. Layers 1–2
and 4–6 retain TokenMLP. Layer 3 (one of the two central layers in the six-layer
stack) uses Lantern's original full-width SwiGLU feed-forward block with hidden
width 1,024. Its attention, normalization, model width, tokenizer, data, seed,
and training recipe match the all-TokenMLP baseline.

| Run | Parameters | Updates / tokens | Time | Best validation loss | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| All six layers TokenMLP | 6,459,904 | 9,000 / 18,429,440 | 308.0 s | 2.242750 | 9.4192 |
| Layer 3 SwiGLU, hidden 1,024 | 7,639,552 | 9,000 / 18,429,440 | 314.0 s | 2.164759 | 8.7125 |

The single full-width FFN adds 1,179,648 parameters. On this one matched-seed
run, it improved validation loss by 0.07799 versus the all-TokenMLP baseline.
This is an exploratory result, not a multi-seed estimate.

The run report contains the validation curve, data subset hashes, and greedy
samples from the best checkpoint. The final best checkpoint is
`results/pretraining/lantern-tokenmlp-middle-ffn-14m-v1/checkpoints/step-00009000.pt`.
