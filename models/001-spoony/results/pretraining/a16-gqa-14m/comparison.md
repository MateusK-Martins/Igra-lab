# RoPE A16 with grouped-query attention

This experiment keeps the standard six-layer, full-width Transformer and SwiGLU backbone, while changing only the A16 attention projections and head layout. Each layer has 4 query heads × 4 features, grouped over 2 KV heads × 4 features (two query heads share each KV head). RoPE is applied independently to each 4D Q/K head. Attention uses causal SDPA with GQA enabled.

| Model | Parameters | Q/K/V projection width | Cache per token/layer | Validation loss | Perplexity | Training time | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard RoPE A16 | 10,375,680 | 16 / 16 / 16 | 32 scalars | 2.170240 | 8.760389 | 215.3s | 573.4 MiB |
| A16 GQA (4Q/2KV) | 10,338,816 | 16 / 8 / 8 | 16 scalars | 2.244303 | 9.433838 | 255.5s | 626.1 MiB |

Both runs used the same tokenizer and processed TinyStories hashes, seed 42, BF16, context length 256, batch size 8, AdamW settings, warmup/cosine schedule, 9,000 updates, and 18,429,440 target tokens. The GQA model has 36,864 fewer parameters (0.36% fewer) and halves the attention cache. Its validation loss is **+0.074063** worse and perplexity is **+0.673449** higher. It was 18.7% slower and used 52.7 MiB more peak allocation in this run.

This single-seed result does not show a benefit from the tested 4-query/2-KV configuration. One plausible explanation is that dividing a 16D query into four 4D heads makes each attention score and retrieved value very narrow; that is an interpretation, not an isolated causal finding.

Final checkpoint: `/home/mateus/Documents/Igra-lab-checkpoints/a16-gqa-14m/step-00009000.pt` (SHA-256 `cab40c7e24322bc6868d038970a19626af11e3bf8645d470b34c8161877a0687`).
