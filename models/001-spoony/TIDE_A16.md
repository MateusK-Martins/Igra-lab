# Tide / A16 hybrid

Pattern [Tide,Tide,A16] repeated twice, six width-384 layers, FFN 1024,
vocabulary 8192 with tied embeddings init std .02. Four unchanged Tide ranged
selective-write/global RMSNorm mixers, two single-head 16D Q/K/V attention
mixers. Bias-free attention and 16-to-384 output; RoPE base 10000; causal full
history, no dropout. Existing GQA runtime configured with one query/KV head
implements A16 exactly; no new attention code. Existing Tide Triton scan unchanged.

12,641,280 actual parameters. Each Tide layer keeps 384 FP32 recurrent values;
each A16 layer caches 32 scalars/token, so total hybrid inference memory includes
growing attention history. External mixed-cache persistence is unchanged.
Same seed 42, TinyStories/tokenizer bytes, BF16, batch 8, context/stride 256,
9000 fresh updates / 18,429,440 target tokens, AdamW3e-4, clip1,
warmup300/cosinefloor.1, validation500/checkpoints1000. No warm start.
Checkpoint link uses /var/tmp/igra-rope-14m/tide-a16 on the root partition because
home lacks room. No existing artifacts removed; this is subject to temporary-file
cleanup policy and should be moved before such cleanup. Metrics/manifests remain
under results/pretraining/tide-a16-14m. Original Tide/Lantern comparison is historical.

This resumes the authorized hybrid experiment after correcting the positional
configuration. Compare against Tide, Lantern, Tide/Lantern and A16+RoPE, while
accounting for different parameter counts/mixer ratios. Single seed only;
no reasoning or retrieval conclusions follow from TinyStories perplexity.
