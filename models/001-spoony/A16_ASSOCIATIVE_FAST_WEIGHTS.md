# A16 associative fast-weight experiment

This branch starts from the six-layer RoPE-A16 run. Each layer keeps the same 16D Q/K/V projections, output projection, and causal K/V cache. It adds a bias-free 384→16 probe projection and a learned scalar β initialized to zero. At width 512, this is 8,193 added parameters per A16 layer; at width 384 it is 6,145 per layer, or 36,870 total. The model therefore has 10,412,550 parameters.

For each token, the regular A16 result is `z_t = sum_i alpha_ti v_i`. The added read is evaluated in factored form:

`f_t = sum_i alpha_ti ((k_i · p_t) / 4) v_i`

where `p_t = W_P x_t`, and `alpha` uses the usual causal A16 scores `q_t · k_i / 4`. This equals querying the temporary operator `sum_i alpha_ti v_i k_i^T` by `p_t`, with the optional extra dot-product scale set to `1/4`. The block output is `x_t + W_O(z_t + beta f_t)`.

The new probe is RoPE-rotated at the current token position, matching the position-aware keys used by the latest standard A16 baseline. β=0 makes the initial block output equal to that baseline. The implementation never constructs or stores a 16×16 matrix: the factorized dot and weighted-value reduction are computed directly. Inference retains only the same 16D keys and 16D values per token per layer (32 scalars total); no extra cache entry or persistent recurrent state is introduced.

Validation uses the pinned TinyStories subset and tokenizer fingerprints from the standard RoPE-A16 run, with seed 42, context/stride 256, batch 8, BF16, AdamW 3e-4, 300 warmup and 9,000 total updates. The run report records the full validation curve, fixed-prompt generations, measured throughput, peak VRAM and final checkpoint hash.
