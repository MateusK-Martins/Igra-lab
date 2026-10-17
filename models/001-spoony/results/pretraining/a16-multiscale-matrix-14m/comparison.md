# Six-layer A16 / matrix-memory comparison

Both runs use the original 14M-class A16 TinyStories recipe: width 384, six total mixer/FFN layers, six SwiGLUs with hidden width 1024, context 256, batch 8, BF16, seed 42, and 9,000 updates (18,429,440 target tokens). Layers alternate A16 then matrix memory, yielding exactly three of each. The multi-scale matrix block uses one learned 0.99 retention scale and fixed 0.90, 0.99, and 0.999 scales. Only the matrix recurrence differs between the two new variants.

| Model | Parameters | Validation loss | Perplexity | Elapsed | Tokens/s | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Pure A16 historical reference | 10,375,680 | 2.17024 | 8.76039 | — | — | — |
| 3× A16 + 3× single-scale matrix | 10,394,520 | 2.23196 | 9.31811 | 213.4 s | 86.4k | 580 MiB |
| 3× A16 + 3× four-timescale matrix | 10,399,140 | 2.20205 | 9.04356 | 220.1 s | 83.8k | 600 MiB |

The multi-scale variant improves validation loss by 0.02991 nats over the single-scale hybrid, while adding 4,620 parameters and about 7 seconds. Both hybrids remain behind the historical pure-A16 run (by 0.06172 and 0.03181 nats, respectively). This is one seed and one budget; it suggests the additional timescales helped within this hybrid, but does not establish that matrix memory is generally inferior.

The learned retention values remained close to their 0.99 initialization. Single-scale layer means were 0.98755, 0.98739, and 0.98694 (feature ranges across layers: 0.98337–0.98980). Multi-scale learned retention means were 0.98903, 0.98902, and 0.98927 (ranges 0.98693–0.99049); its other three scales stayed fixed at 0.90, 0.99, and 0.999. Each per-variant `training-report.json` records the full validation curve, fixed-prompt generations, GPU/runtime measurements, and checkpoint hash. Final checkpoints are stored under `/home/mateus/Documents/Igra-lab-checkpoints/`.
