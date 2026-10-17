# A16 associative fast weights — comparison

This is standard six-layer RoPE A16 plus the proposed temporary associative fast-weight readout. It preserves the 16D Q/K/V projections and ordinary K/V cache. For each current query, the implementation computes the factored equivalent of the 16×16 operator; no matrix is stored or allocated as a persistent state. The independent 16D probe is RoPE-positioned, and the learned scalar beta starts at zero.

| Run | Parameters | Updates / tokens | Validation loss | Perplexity | Wall time | Peak CUDA allocated |
|---|---:|---:|---:|---:|---:|---:|
| Standard RoPE A16 | 10,375,680 | 9,000 / 18,429,440 | 2.170240 | 8.760389 | 215.3s | 573.4 MiB |
| A16 + associative fast weights | 10,412,550 | 9,000 / 18,429,440 | 2.157974 | 8.653590 | 251.4s | 630.9 MiB |

The baseline and experiment use matching TinyStories/tokenizer fingerprints, seed 42, BF16, context 256, optimizer/schedule, and validation protocol. The experiment improved loss by 0.012266 (0.107 perplexity), while runtime increased 16.8% and peak allocation increased 57.5 MiB. This is a modest single-seed result, not a robust superiority claim.

Learned beta by A16 layer: +0.0485, +0.1025, -0.1120, -0.0900, +0.0630, +0.0964. All six layers moved away from zero, indicating the added pathway was used, though this alone does not establish what information it contributes.

Final checkpoint: `/home/mateus/Documents/Igra-lab-checkpoints/a16-associative-fast-weights-14m/step-00009000.pt` (SHA-256 `d590f8f8f8d3efa80d7991b359ea700e2b4829139d9ad027478eef50a7a21bb2`).
