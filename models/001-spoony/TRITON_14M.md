# Tide 14M Triton retraining

Fresh seed-42 retraining of the original Tide recipe: width384, six layers,
SwiGLU1024, vocabulary8192, context256, batch8, BF16, 9000 optimizer updates.
13,774,080 parameters; ranged retention/write initialization and global RMSNorm
are unchanged. No trained weights are used. Data/tokenizer hashes were checked
against results/pretraining/fluxv-init-range/manifest.json and match exactly.

Only the CUDA FP32 affine scan backend changes. CPU/FP64 keep the original
PyTorch reference. The Triton scan accumulates in FP32 and implements first-order
backpropagation and optional constant-size initial state. Parallel regrouping and
GPU fused arithmetic can change rounding; training need not be bit-identical.

Artifacts are isolated under results/pretraining/tide-triton-14m/. The existing
benchmark and run_baseline scripts record timing, CUDA peak allocation, validation
curve and fixed greedy generations. Historical Tide loss2.010153/PPL7.464462.
The benchmark discards its model; training starts in a fresh Python process.

## Completed result

| Backend | Validation loss | Perplexity | Training seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Historical PyTorch | 2.010153 | 7.464462 | 378.97 | 0.999 |
| Triton | 2.010163 | 7.464536 | 270.18 | 0.726 |

Both processed 18,429,440 target tokens over 9000 updates. Artifact hashes and
resolved training settings match. Training time includes validation and checkpoint
saving; fixed-prompt generation follows the timed training phase. Throughput over
the full timed run is approximately 68,213 target tokens/second (historical48,630).
The saved benchmark measures mean whole-model update time28.87ms versus40.46ms.
Training is approximately1.40x faster with27.3% less peak allocated CUDA memory.
These timings compare historical runs on the same GPU, not interleaved hardware
trials. The loss difference is0.00000983; this run provides no evidence of a
material modeling-quality change. Floating-point regrouping/fusion can explain
numerical differences without changing the mathematical recurrence.

The best saved checkpoint is step9000. Full curve and fixed-prompt greedy samples
are in results/pretraining/tide-triton-14m/training-report.json. Samples still show
repetition; matching perplexity is not evidence of reasoning ability. Checkpoints
remain ignored by Git. Sleep/idle inhibition covered benchmark and training and
ended automatically when the process exited.

Verification:725 tests passed, including25 CUDA Triton tests and a complete-model
BF16 logits/parameter-gradient comparison; Ruff lint and formatting passed.
