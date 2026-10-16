# SMA4 with independent keys

Branch: model/001-spoony-sma4-key-14m. Parent: model/001-spoony-sma4-14m.
The original A16/A4/SMA4 branches remain unchanged.

Per token, q = W_Q h + b_Q, k = W_K h (both four features),
M = reshape(W_M h + b_M, 4, 4). Causal scores are q_t^T k_i / 2;
alpha = softmax(scores); z_t = sum_i alpha_ti (M_i q_t);
output = W_O z_t + b_O. Only the final four-dimensional aggregate is decoded.
The learned scoring vector u is removed. The independent key is bias-free,
as in A4. Existing SMA query/matrix/output biases and FP32 accumulation remain.
Each layer caches only keys [B,T,4] and matrices [B,T,4,4], 20 scalars per token.
Both parts append atomically after validation and share one count. Keyed matrix
snapshots use format 3; existing KV/standalone-matrix formats 1/2 still load.
Old SMA weights do not load strictly as this architecture: it is a fresh run.

No positional rotation, no attention dropout, unrestricted causal cache, exactly
as in the prior controlled A16/A4/SMA4 suite. Same full-width Lantern backbone:
six width-384 layers, SwiGLU 1024, vocabulary 8192, tied embedding init std .02.
Actual parameters: 10,295,160 (+9,192 versus SMA4: six bias-free 384x4 key
projections replace six four-element scoring vectors). No FFN padding.
Unchanged-backbone initialization remains identical across modes. Different
attention projections retain standard PyTorch initialization; attention RNG
consumption is isolated from embeddings/FFNs. This does not promise identical
initial matrix/output parameters between the two attention mechanisms.

The same seed 42, tokenizer/data bytes, ordering, context/stride 256, batch 8,
BF16, AdamW 3e-4, betas .9/.95, epsilon 1e-8, weight decay .01, clipping 1,
300-step warmup, cosine floor .1, validation every 500, checkpoints every 1000,
and 9000 updates / 18,429,440 target tokens are preserved. Fresh initialization;
benchmark training is discarded in a separate process. No environment changes.
Output directory: results/pretraining/sma4-key-14m.

Tests cover reference outputs/all parameter gradients, FP64 gradcheck, causal
future-gradient isolation, unchanged backbone initialization and exact counts,
single-token/chunked cached decoding, cache snapshots and cross-device reads,
BF16 CPU/CUDA forward/backward, and independent-key selection of fixed matrices.
Invalid keyed writes leave both cache tensors unchanged. Old vector-scoring
SMA weights are rejected; the new cache stores neither full hidden states nor
conventional values. Full suite and Ruff must pass before training.

This tests separate scoring from matrix responses. The same q still controls
score sharpness and response magnitude; that coupling is deliberately unchanged.
Key cache/parameters increase slightly, and attention initialization is not an
exact common-parameter match, so any loss change is not a perfectly isolated
causal estimate of selection/content coupling. Results are single-seed TinyStories.

## Completed run

All 519 tests and Ruff lint/format checks passed before training. The run completed all 9000 updates / 18,429,440 target tokens. Initial/final logits and gradients were finite. Training-data/tokenizer fingerprints and initial shared-backbone weight hash match the previous SMA4 run. Current runtime source hashes match the frozen manifest.

| Model | Parameters | Loss | Perplexity | Minutes | Target tokens/s | Peak GiB | Cache scalars/token/layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A16 | 10,375,680 | 2.271129 | 9.690332 | 3.69 | 83,307 | 0.560 | 32 |
| A4 | 10,265,088 | 2.488885 | 12.047835 | 3.91 | 78,573 | 0.571 | 8 |
| SMA4 | 10,285,968 | 2.677201 | 14.544328 | 4.58 | 67,063 | 0.610 | 16 |
| SMA4 + independent keys | 10,295,160 | 2.508731 | 12.289329 | 3.96 | 77,493 | 0.620 | 20 |

Independent-key loss minus original SMA4: -0.168470; minus A4: +0.019846. These are observed single-seed differences, not proof of the mechanism responsible. Keys separate token selection from matrix content, but query magnitude still changes both score sharpness and response amplitude.

Fixed-batch gradient norm before clipping: 1.962352 initially, 4.229872 finally. The optimizer clips at 1.0. Layer output RMS, attention entropy, query RMS and matrix singular values are in the probes. Peak memory/time refer to training including validation/checkpoint saves, excluding probes, benchmark and samples.

## Fixed-prompt sample

Once upon a time, there was a little girl named Lily. She loved to play outside in the garden. One day, she saw a big, scary dog. She wanted to help her.
Lily asked her mom, "Can I play with you?" Her mom said, "Yes, I can help you."
Lily was so happy to see her mom. She said,

All fixed prompts and the complete validation curve are in training-report.json. Samples follow the established greedy/best-saved-checkpoint protocol; table metrics use update 9000. Large checkpoints and datasets remain ignored.

The trained checkpoint also passed full-sequence versus 7-token cached chunks
on the fixed 21-token prompt: maximum logit difference 1.24e-05
in FP32 and 5.92e-14 in FP64.
See trained-cache-equivalence.json. This supplements the pre-training reference,
causal and snapshot tests; it does not require bitwise BF16 chunk equivalence.
