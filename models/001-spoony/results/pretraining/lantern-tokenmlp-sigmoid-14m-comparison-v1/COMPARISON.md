# Lantern 14M sigmoid-gate TokenMLP comparison

All four rows use the same 8192-token tokenizer, prepared TinyStories train/validation artifacts, 18,429,440 target tokens, 9,000 updates, seed 42, context 256, batch 8 and BF16. The two sigmoid variants differ only in the second gate: a learned token table versus the shared down-projected current hidden state. Training time includes validation and checkpoint saves.

| Model | Parameters | Validation loss | Perplexity | Training time |
|---|---:|---:|---:|---:|
| Original Lantern | 12,440,064 | 2.018320 | 7.525675 | 431.33 s |
| Shared tanh-matrix TokenMLP | 6,459,904 | 2.242750 | 9.419195 | 308.00 s |
| Sigmoid learned pair | 6,459,904 | 2.285694 | 9.832505 | 287.97 s |
| Sigmoid context gate | 5,935,616 | 2.279891 | 9.775617 | 285.28 s |

The context gate is 524,288 parameters smaller and improves validation loss by 0.005802 nats against the learned-pair variant in this single-seed run. Both sigmoid variants have higher validation loss than the previous shared tanh-matrix TokenMLP and the original Lantern. The small context-versus-learned gap needs repeated seeds before treating it as a reliable difference.

Run reports with validation curves, greedy samples, hardware, peak memory and manifests:

- [Learned-pair report](../lantern-tokenmlp-sigmoid-learned14m-v1/training-report.json)
- [Context-gate report](../lantern-tokenmlp-sigmoid-context14m-v1/training-report.json)
