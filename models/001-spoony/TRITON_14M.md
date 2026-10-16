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
