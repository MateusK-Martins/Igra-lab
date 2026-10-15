# Independent write gate + squared-retention No-K

Fresh controlled combination: width384, six layers, SwiGLU1024, vocabulary8192.
S = g*S_previous + i*tanh(W_V*x)
P = g²*P_previous + 1
output = sigmoid(W_Q*x)*gain*S/sqrt(P+1e-6)
The existing residual wrapper adds x once. No internal RMS normalization.
g is the original token-dependent retention, initialized0.99; independent
i is a token-dependent sigmoid projection initialized0.5 (zero weights/bias).
Denominator tracks squared retention only, not i². This preserves the prior
denominator ablation and allows write suppression to reduce read amplitude.
Gain starts at1. Existing parallel affine scan packs S/P; FP32 accumulation
under BF16. Constant external cache: two384-element vectors,3KiB/sample/layer.
Distinct cache ID isolates gated and ungated checkpoints. Same original FFN.

13,774,080 parameters: +887,040 versus squared-retention No-K, same count as
RMS-normalized write-gate No-K and original FluxV. Unchanged parameters retain
identical seed42 initialization; extra projection construction preserves RNG.
Same pinned TinyStories30k/256 documents, tokenizer, seed42, context256,batch8,
9000updates, AdamW3e-4, warmup300, BF16, validation500, target18429440tokens.
No warm start, no other architecture changes. Single-seed limitations apply.

666 tests passed, including independent FP64 sequential math/gradients,
write-gate initialization/suppression, denominator independence, causality,
gradcheck, chunk/single-token cache save/load, constant memory, long sequences,
residual once and BF16 FP32-state finite gradients. Ruff checks pass.

## Completed controlled run

All models use the same data/tokenizer fingerprints, seed42 and9000 updates /18429440 actual target tokens. Historical baseline timing is separately measured.

| Model | Parameters | Loss | PPL | Minutes | Tokens/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| No-K RMSNorm | 12,887,040 | 2.123176 | 8.357640 | 6.84 | 44938 | 0.940 |
| No-K squared retention | 12,887,040 | 2.112602 | 8.269727 | 10.62 | 28926 | 1.285 |
| Write gate + RMSNorm | 13,774,080 | 2.090822 | 8.091565 | 6.28 | 48876 | 0.999 |
| Write gate + squared retention | 13,774,080 | 2.076412 | 7.975798 | 11.03 | 27846 | 1.347 |

Same-capacity comparison against write gate + RMSNorm isolates normalization. Comparing against squared retention alone includes +887040 projection parameters. Single seed cannot establish general superiority or reasoning/retrieval. The retention-only denominator does not compensate for input-gate suppression or correlated/nonzero-mean writes.

### Initial/final held-out probes

initial: gradient norm 2.875980; all gradients finite: True.
- body.1.branch: mixer RMS 0.20370, S RMS 3.31942, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
- body.5.branch: mixer RMS 0.29238, S RMS 5.34529, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
- body.9.branch: mixer RMS 0.38917, S RMS 7.75592, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
- body.13.branch: mixer RMS 0.50959, S RMS 10.75299, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
- body.17.branch: mixer RMS 0.59915, S RMS 12.96505, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
- body.21.branch: mixer RMS 0.71921, S RMS 15.18263, P mean 49.95898, write gate mean 0.50000, token std 0.00000.
final: gradient norm 2.236424; all gradients finite: True.
- body.1.branch: mixer RMS 0.41786, S RMS 1.95720, P mean 6.26729, write gate mean 0.50394, token std 0.25946.
- body.5.branch: mixer RMS 0.54380, S RMS 1.60031, P mean 5.07575, write gate mean 0.55430, token std 0.23402.
- body.9.branch: mixer RMS 0.51071, S RMS 2.97993, P mean 9.92165, write gate mean 0.51491, token std 0.22414.
- body.13.branch: mixer RMS 0.51123, S RMS 4.48245, P mean 13.65483, write gate mean 0.51130, token std 0.20356.
- body.17.branch: mixer RMS 0.51344, S RMS 4.02449, P mean 11.11080, write gate mean 0.54170, token std 0.19687.
- body.21.branch: mixer RMS 0.46166, S RMS 3.95690, P mean 11.27536, write gate mean 0.51069, token std 0.18575.

### Fixed-prompt generations

Best saved checkpoint: checkpoints/step-00009000.pt. Loss table uses final-budget weights.

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was scared of the dog and wanted to play with her.
> Lily's mom saw her and said, "Don't worry, Lily. The dog is just playing. He is just playing. He is not

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not happy. It wanted to play with Tom and Sam.
> "Can I play with you?" Tom asked. "I want to play with you."
> "No, Tom, you can't play with my ball. It is mine. You have your own toys. You have your own toys. You have your

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy, but he could not find it. He looked under the bed, in the closet, and in the toy box, but he could not find it.
> Then, he saw a big dog. The dog was not nice. The dog wanted to play with the

### Interpretation

The combination lowers final loss by0.01441 versus the same-size RMS-normalized write-gate model (about1.43% lower perplexity), and by0.03619 versus squared retention alone. It is the strongest completed No-K result so far. The two-by-two loss comparison suggests a favorable interaction: the squared-retention loss improvement is0.01057 without the input gate and0.01441 with it, an extra0.00384 improvement. This small interaction needs repeated seeds before drawing a robust conclusion.

Write-gate token variation became nonzero in every layer (mean per-feature token std0.186–0.259), so the gate did not remain a constant multiplier. This is evidence of learned input dependence, not proof of long-term selective memory. Gradients and sampled state/readout activations stayed finite. Runtime was about1.76x the historical RMS-normalized write-gate run; the unchanged larger packed affine scan computes and stores the additional denominator. Generations still contain repetition and story inconsistencies.
