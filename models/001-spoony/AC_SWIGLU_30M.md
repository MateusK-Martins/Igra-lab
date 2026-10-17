# A16-conditioned SwiGLU, 30M scale

Scale the best A16 AC-SwiGLU 14M configuration, without Tide or gMLP.
14 layers, width 512, SwiGLU 1,152, tied 8,192-token embedding/head; actual
29,705,216 parameters. Increased depth reaches30M while preserving the
previous 30M width/FFN. Compared with9-layer 30M models, depth is different.
The original AC runtime is byte-for-byte unchanged.

Each layer computes RoPE A16 on RMS-normalized input, attention residual,
pre-FFN RMSNorm and SwiGLU, modulated by γ = 1 + 0.1 × tanh(W_C z), then its
FFN residual and existing final RMSNorm. W_C: 16→1,152, bias-free and zero-initialized.
Q/K/V remain 16D; the KV cache stores 32 scalars per token per layer, no new persistent state.
Conditioning parameters:14*16*1152=258,048. No trained-weight warm start.

Use the established30M two-stage pipeline unchanged:200kTinyStories,
30kSQuAD, same saved tokenizer, seed42, context512,batch8,BF16,
6,500 mixed updates at LR 3e-4 with 200 warmup steps; then 6,500 QA-only updates at LR 3e-5
with 100 warmup steps with a fresh optimizer initialized from this run's own mixedfinal.
Both use AdamW with β=(0.9, 0.95), weight decay 0.01, gradient clipping 1.0 and cosine floor 0.1.
Each stage consumes26,624,000 padded positions. Audit exact supervised tokens
and data/settings hashes against historical30M manifests before training.
QA prompts/padding are masked; padded positions differ from target tokens.

Restore the saved tokenizer and regenerate datasets only; no tokenizer retrain.
Prepared hashes must match before starting. Existing training, SFT and evaluation
scripts are reused. Evaluate TinyStories/QA loss and PPL, greedy256-question
EM/F1 and fixed story prompts both before and after QA. Run BF16 gradient and activation
probes and FP32/FP64 full and cached comparisons on both final checkpoints.

One seed; increased depth and parameter differences limit architecture claims.
The 14M and 30M runs use different corpora and tokenizers, so their perplexities cannot be
compared directly. The earlier14M conditioning gain over its matched
normalization control was negligible; this scale-up does not by itself isolate
conditioning from capacity. Lower QA loss is not proof of reasoning/retrieval.

## Completed comparison

Both stages completed 6,500 updates and 26,624,000 padded input positions each. Source/artifact/settings hashes, exact supervised exposure and own-checkpoint QA initialization passed final audits. All 578 tests and Ruff passed before training.

| Model | Parameters | Mixed min | Story loss / PPL before QA | QA loss / PPL before QA | EM / F1 % |
|---|---:|---:|---|---|---|
| Lantern | 29,565,952 | 20.48 | 2.055921 / 7.8140 | 6.080587 / 437.29 | 0.0000 / 1.1896 |
| Tide/A16 | 26,527,744 | 11.23 | 2.051892 / 7.7826 | 6.139653 / 463.89 | 0.0000 / 0.8210 |
| A16 + AC-SwiGLU | 29,705,216 | 17.84 | 2.195472 / 8.9842 | 6.257823 / 522.08 | 0.0000 / 0.2604 |

| Model | QA min | Story loss / PPL after QA | QA loss / PPL after QA | EM / F1 % | Story PPL ratio |
|---|---:|---|---|---|---:|
| Lantern | 21.26 | 2.564399 / 12.9928 | 5.078574 / 160.54 | 0.7812 / 2.3663 | 1.6628 |
| Tide/A16 | 11.69 | 2.646863 / 14.1097 | 5.223213 / 185.53 | 0.0000 / 0.9916 | 1.8130 |
| A16 + AC-SwiGLU | 18.53 | 2.754155 / 15.7078 | 5.251683 / 190.89 | 0.3906 / 1.2689 | 1.7484 |

Peak allocated CUDA GiB (mixed,QA): 2.887, 2.887. Padded positions/s: 24,869, 23,950. Timings include loading/validation/saving; final evaluations/probes/generations are separate.

All 14 layers use A16 plus conditioned SwiGLU. Historical30M Lantern and Tide/A16 have9layers and different mixers/norm placement. These are pipeline references, not matched controls for conditioning. A future matched, unconditioned 30M run would be required to isolate conditioning.

## Stability and conditioning

Initial: BF16 gradient norm before clipping 3.941603; sampled activations/gradients finite. Gamma mean absolute deviation by layer: body.0=0.000000, body.2=0.000000, body.4=0.000000, body.6=0.000000, body.8=0.000000, body.10=0.000000, body.12=0.000000, body.14=0.000000, body.16=0.000000, body.18=0.000000, body.20=0.000000, body.22=0.000000, body.24=0.000000, body.26=0.000000.

After mixed: BF16 gradient norm before clipping 1.455807; sampled activations/gradients finite. Gamma mean absolute deviation by layer: body.0=0.000720, body.2=0.003349, body.4=0.002448, body.6=0.003083, body.8=0.002571, body.10=0.004937, body.12=0.005240, body.14=0.002735, body.16=0.004066, body.18=0.004511, body.20=0.005923, body.22=0.006378, body.24=0.004872, body.26=0.006515.

After QA: BF16 gradient norm before clipping 7.439228; sampled activations/gradients finite. Gamma mean absolute deviation by layer: body.0=0.000722, body.2=0.002830, body.4=0.001781, body.6=0.002147, body.8=0.002388, body.10=0.004261, body.12=0.004853, body.14=0.002527, body.16=0.003501, body.18=0.004466, body.20=0.005438, body.22=0.005921, body.24=0.004709, body.26=0.005925.

Trained 141-token full versus7-token cached chunks passed FP32/FP64 for both final checkpoints. KV storage32scalars/token/layer,448 across 14 layers; no extra conditioning cache.

Before QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily's mom told her they were going to the park to play. Lily was so excited to go on the swings and slide down the slide.
> When they got to the park, Lily saw a big slide. She wanted to go on it

After QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She was very happy and loved to play with her friends.
> As they played, Lily's mom asked her to play with her toys. Lily didn't want to play with her toys, but her mom said, "I'm sorry, Lily. I don't want to play with you."
> Her mom smiled and
