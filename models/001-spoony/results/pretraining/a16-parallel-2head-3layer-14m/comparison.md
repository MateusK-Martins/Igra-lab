# Three layers of two standard A16 heads

Each of the two attention heads in a layer is a full, independent standard A16 module: separate 16D Q/K/V projections, causal attention, RoPE, and a 16→384 output projection. The two 384D outputs are summed with a fixed `1/sqrt(2)` scale, then added to the residual once. One SwiGLU follows each combined attention block.

To reach approximately 14M parameters with only three FFNs, SwiGLU hidden width is 3,072 (the standard six-layer A16 baseline uses 1,024). Exact model count: **13,912,320 parameters**. This means the controlled comparison changes layer depth, the grouping of attention modules, and FFN capacity.

| Model | Parameters | Residual layers | A16 modules | SwiGLU hidden | Validation loss | Perplexity | Time | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard RoPE A16 | 10,375,680 | 6 | 6 | 1,024 | 2.170240 | 8.760389 | 215.3s | 573.4 MiB |
| 3-layer, 2-head A16 | 13,912,320 | 3 | 6 | 3,072 | 2.173665 | 8.790439 | 247.3s | 623.0 MiB |

Both runs used the exact same processed data/tokenizer hashes, seed 42, BF16, context 256, batch size 8, optimizer/scheduler, and 9,000 updates / 18,429,440 target tokens. Loss differs by +0.003424; PPL differs by +0.030050. The new model took 14.8% longer and used 49.6 MiB more peak allocated memory. Its per-layer cache stores two independent A16 K/V streams (64 scalars/token/layer), totaling the same 192 scalars/token across three layers as six standard A16 layers.

The validation result is close to the baseline, but this single-seed run cannot attribute the result to head grouping alone because the FFN width and number of residual stages also changed.

Final checkpoint: `/home/mateus/Documents/Igra-lab-checkpoints/a16-parallel-2head-3layer-14m/step-00009000.pt` (SHA-256 `3db1a8d0d4d2467105bbfb3a45a2b2a24e33e0ed15a552de4b94532e0149953c`).
