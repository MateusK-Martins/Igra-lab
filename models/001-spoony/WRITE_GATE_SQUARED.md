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
