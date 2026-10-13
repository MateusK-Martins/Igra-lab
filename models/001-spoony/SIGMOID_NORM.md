> **Historical experiment:** current Ember configuration and results are in
> [DECAY_EXPERIMENTS.md](DECAY_EXPERIMENTS.md).

> **Ember:** this model now lives on `model/001-spoony-ember`. Historical
> recipe commits and reports retain their original branch identities.

# Sigmoid Q/V and normalization experiments

Two fresh seed-42, 9,000-update runs use the same prepared TinyStories artifacts,
batch order, context 256, BF16, optimizer and schedule as the previous cumulative
model. Both apply sigmoid to Q and V projections, retaining tanh on K. Neither
run initializes from a trained checkpoint. Each stage has its own recipe commit,
manifest, benchmark, saved checkpoints and fixed-prompt greedy samples.

Stage 1: retain both trainable RMSNorms inside the cumulative mixer, to measure
sigmoid Q/V separately. Outputs: results/pretraining/attentionless-sigmoid-rms-v2/.

Stage 2 (implemented): replace only the two norms inside the cumulative mechanism
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

## Stage 2 recipe

The configured cumulative mixer now selects normalization="soft_rms". The
original RMSNorm option remains available. Only the two internal norms change;
outer RMSNorms are preserved. The soft variant uses no epsilon because its
denominator already has a unit floor, and no learned gains. Parameter count is
12,882,432 versus stage 1's 12,887,040. New outputs use
results/pretraining/attentionless-sigmoid-soft-rms-v3/.

509 tests passed before the battery interruption, including both norm choices'
formula/gradient references, causality, cached continuation and CUDA precision
checks, plus soft norm gradcheck, identity Jacobian at zero, non-amplification,
dtype preservation and unchanged common initial projection weights. Lint and
format checks passed again after resuming. No stage-2 training had started before
the interruption; the benchmark was rerun successfully after resuming.

## Stage 2 completed — 2026-10-13

Recipe commit `9b7f253`. The fresh run completed in 265.14 seconds (4m 25s),
including validation and saving. It consumed 18,429,440 target tokens in 9,000
updates, with the exact same prepared artifact hashes and training settings as
stage 1 and the original cumulative run. Final validation loss was 2.718405,
perplexity 15.156136. The final checkpoint step-00009000.pt is best among nine
saved checkpoints; it was loaded for generation and matched its saved manifest.

| Variant | Parameters | Final validation loss | Perplexity | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| Original Q/V + RMSNorm | 12,887,040 | 2.659388 | 14.287542 | 290.50 |
| Sigmoid Q/V + RMSNorm | 12,887,040 | 2.636655 | 13.966409 | 324.14 |
| Sigmoid Q/V + soft RMS | 12,882,432 | 2.718405 | 15.156136 | 265.14 |

Sigmoid Q/V gave a small loss improvement in this seed. The proposed gentler
norm was worse by 0.081750 nats versus sigmoid Q/V with RMSNorm. Its guaranteed
non-amplification did not translate into better validation loss. Changing the
denominator and removing learned gains are bundled in this comparison; these
results do not isolate their individual effects. Both variants still produce
repetitive and inconsistent generations. These are preliminary single-seed
results, not proof of general improvement or inferiority. Stage 1 and stage 2
ran on opposite sides of a battery interruption, so measured wall times should
not be treated as a controlled isolated norm speed comparison.

The current model configuration retains the requested soft norm. To choose the
better-scoring sigmoid/RMS variant, select normalization="rms" in the cumulative
mixer config. To load an existing trained model, use its recorded recipe commit
and manifest; do not reinterpret historical weights under a changed recipe.
Checkpoints and processed artifacts remain ignored and local. Code and reports
are committed locally, not pushed.

Full stage-1 results:
[training-report.json](results/pretraining/attentionless-sigmoid-rms-v2/training-report.json).
Full stage-2 results:
[training-report.json](results/pretraining/attentionless-sigmoid-soft-rms-v3/training-report.json).
The machine-readable comparison is
[norm-comparison.json](results/pretraining/attentionless-sigmoid-soft-rms-v3/norm-comparison.json).
