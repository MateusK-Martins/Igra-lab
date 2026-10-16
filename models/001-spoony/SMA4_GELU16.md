# SMA4 independent keys with nonlinear 16D contributions

Branch model/001-spoony-sma4-gelu16-14m, based on the completed independent-key
SMA4. Original experiment branches remain unchanged. Historical reports inherited
from the parent are not results for this variant.

q_t = W_Q h_t + b_Q (4D); k_i = W_K h_i (4D);
M_i = reshape(W_M h_i + b_M, 4, 4).
alpha_ti = causal_softmax(q_t^T k_i / 2).
r_ti = M_i q_t (4D).
f_ti = GELU(W_P r_ti + b_P) (16D).
z_t = sum_i alpha_ti f_ti (16D).
o_t = W_O z_t + b_O (384D).

W_P is one biased 4-to-16 projection per layer, shared across tokens and queries.
GELU uses PyTorch's exact default (approximate='none'). It acts on each individual
response before weighted aggregation. It is not moved after the sum. The output
projection changes from 4-to-384 to 16-to-384. Matrix/key/query sizes and selection
scores are unchanged. Only the final aggregate is projected to the residual width.
The cache still contains keys and 4x4 matrices, 20 scalars/token/layer; projected
responses are computed per query and never cached. Snapshot format remains 3.
The parent checkpoint is structurally incompatible; training starts fresh.

All response/projection/GELU/softmax/aggregation operations accumulate in FP32
under BF16, preserving FP64 for references. The explicit training temporary now
has [B,T,S,16] elements rather than [B,T,S,4]; memory and compute grow accordingly.
No scan optimization or environment changes. Learned layers use standard PyTorch
initialization. Shared-backbone initial values are identical to prior variants;
attention initialization is not an exact common-parameter match (changed output
projection size changes the subsequent attention-only RNG draws).

Actual total: 10,323,288 parameters, +28,128 over independent-key SMA4:
six layers times (12*384 additional output weights + 4*16+16 expansion parameters).
Backbone remains six width-384 layers, FFN 1024, tied vocabulary-8192 embeddings
with std .02. No RoPE, no dropout, one attention module/head/layer, full causal cache.
Same TinyStories/tokenizer bytes, seed 42, loader order, batch 8, context/stride 256,
BF16, AdamW 3e-4 / (.9,.95) / eps 1e-8 / decay .01, clipping 1, warmup 300,
cosine floor .1, validation every 500, saves every 1000, 9000 fresh updates /
18,429,440 target tokens. Benchmark updates are discarded in a separate process.
Results: results/pretraining/sma4-gelu16-14m.

529 tests passed before training, including independent sequential references
for outputs/all gradients, FP64 gradcheck, an explicit before-versus-after-sum
counterexample, exact parameter count, preserved backbone initialization, causal
future-gradient isolation, single-token/chunked cache equivalence and snapshots,
full-model cached logits, BF16 CPU/CUDA stability and long/zero-input behavior.
Ruff lint and formatting passed. Benchmark memory before full training.

This tests nonlinear per-contribution expansion, not a 16D query or a larger
matrix memory. The same four-dimensional query still scores and reads memory.
The expanded nonlinear features can encode statistics not available from only
the mean 4D response; they do not create information absent from the original
responses. Comparison is single-seed and introduces additional parameters and
computation; it cannot prove general architectural superiority.

## Completed results

All 9000 updates / 18,429,440 target tokens completed. Dataset/tokenizer artifact hashes and initial shared-backbone weights match the prior run. Runtime sources remained identical to the frozen manifest. Initial/final logits and gradients are finite.

| Model | Parameters | Loss | Perplexity | Minutes | Target tokens/s | Peak GiB | Cache scalars/token/layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| A16 | 10,375,680 | 2.271129 | 9.690332 | 3.69 | 83,307 | 0.560 | 32 |
| A4 | 10,265,088 | 2.488885 | 12.047835 | 3.91 | 78,573 | 0.571 | 8 |
| SMA4 | 10,285,968 | 2.677201 | 14.544328 | 4.58 | 67,063 | 0.610 | 16 |
| SMA4 + independent keys | 10,295,160 | 2.508731 | 12.289329 | 3.96 | 77,493 | 0.620 | 20 |
| SMA4 keys + GELU16 contributions | 10,323,288 | 2.503748 | 12.228245 | 6.14 | 49,987 | 0.996 | 20 |

Loss difference versus independent-key SMA4: -0.004983. This measures the combined effect of nonlinear per-response expansion and a wider output decoder under the same data/update budget. It does not isolate nonlinearity from decoder width or prove a general advantage. Cache cost is unchanged; training compute and memory increase.

Fixed-batch gradient norm before clipping: 1.596952 initially → 3.934815 finally. All parameters had finite gradients; configured clipping remains 1.0. See initial/final-probe.json for layer RMS, entropy, query magnitude and matrix singular values.

## Fixed-prompt generation

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to the park with her mom. She saw a big tree with a big, red ball. She picked it up and started to climb it.
Lily was very happy to see the birds. She loved to play with her toys and play with her

All fixed prompts and the full validation curve are in training-report.json. Generations use the best saved validation checkpoint and the existing greedy protocol; table metrics use the final update. Times include validation/checkpoint saving, excluding benchmark, probes and generations. Datasets/checkpoints remain ignored.

Final trained-cache verification passed on the fixed 21-token prompt, using
7-token chunks: maximum FP32 logit difference 1.34e-05,
FP64 3.38e-14.
See trained-cache-equivalence.json. The modest loss improvement is not enough
to establish a reliable gain from this single seed, particularly given higher
training cost. A4 still has lower validation loss with a smaller cache.
