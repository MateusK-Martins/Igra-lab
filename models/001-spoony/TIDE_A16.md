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

## Completed results

All 730 tests and Ruff passed before training. Completed 9000 updates /
18,429,440 target tokens, final loss 2.025097, perplexity
7.576844. Training including validation/saving took 249.338s
(4.16 minutes), throughput 73,914 target tokens/s,
peak allocated CUDA memory 718,273,536 bytes
(0.669 GiB).

Reference losses: pure Tide 2.010153; Lantern 2.018320; prior Tide/Lantern
1.983780; A16+RoPE 2.170240. These comparisons change mixer
ratios and parameter counts, so they do not isolate only attention width.
Same data/tokenizer hashes as the original hybrid; runtime source hashes match
the frozen manifest. Initial and final BF16 probes have finite gradients;
norms before clipping 2.556218 and
1.719267. Gate/state statistics remain in probes.

Trained full-versus-cached decoding passed: FP32 maximum logit difference
9.54e-06, FP64
2.58e-14. Cache tests also cover
mixed attention/recurrent snapshot restoration and future-gradient isolation.
Complete validation curve and all fixed greedy samples are in training-report.json.
Samples use the best saved checkpoint; final metrics use update 9000.

Fixed generation:

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very scary and Lily was scared.
Lily's mom told her to be careful and not to hurt the dog. She said, "Don't be scared, Lily. The dog is friendly and wants to
