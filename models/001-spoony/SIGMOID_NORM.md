# Sigmoid Q/V and normalization experiments

Two fresh seed-42, 9,000-update runs use the same prepared TinyStories artifacts,
batch order, context 256, BF16, optimizer and schedule as the previous cumulative
model. Both apply sigmoid to Q and V projections, retaining tanh on K. Neither
run initializes from a trained checkpoint. Each stage has its own recipe commit,
manifest, benchmark, saved checkpoints and fixed-prompt greedy samples.

Stage 1: retain both trainable RMSNorms inside the cumulative mixer, to measure
sigmoid Q/V separately. Outputs: results/pretraining/attentionless-sigmoid-rms-v2/.

Stage 2 (planned): replace only the two norms inside the cumulative mechanism
with feature-wise `x / sqrt(1 + mean(x**2))`, without learned gain. This smooth
compression has denominator at least one, so it does not amplify small signals;
its derivative at zero is one rather than RMSNorm's 1/sqrt(epsilon). At large
magnitudes it approaches unit RMS. It still compresses magnitude, and improvement
in training is a hypothesis. Outer RMSNorms remain unchanged. Removing the two
internal learned gains removes 4,608 parameters (0.036%); record this explicitly.

The second run must start from scratch, use a fresh output directory and keep
all other training settings fixed. Norms operate per token across features;
sequence aggregation remains causal cumsum along dim=1 with FP32 accumulation.

## Stage 1 completed

Recipe commit `2b32df1`. All 495 tests and Ruff passed. The fresh sigmoid Q/V
plus RMSNorm run completed in 324.14 seconds (5m 24s), including validation
and saving. Final validation loss 2.636655, perplexity 13.966409. Final checkpoint
step-00009000.pt is best among saved checkpoints and was loaded for generation;
its settings match the saved manifest. It consumed exactly 18,429,440 tokens.
Compared with the original cumulative model (loss 2.659388), the loss improvement
is small (0.022733 nats); this is not a multi-seed conclusion.
