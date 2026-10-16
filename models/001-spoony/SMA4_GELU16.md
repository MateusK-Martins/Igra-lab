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
