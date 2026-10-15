# Specialized No-K mechanism ablations — retention

Standard reference remains global-RMSNorm write-gated No-K13,774,080 parameters.
This branch is one fresh controlled experiment, not promoted to standard.
Historical selective-head results remain in SELECTIVE_HEADS.md.

A/global: specialized retention(.90,.95,.98,.995),write(.20,.10,.05,.02),
global RMSNorm over384.
B/retention: retention(.90,.95,.98,.995),write(.50,.50,.50,.50),
four independent96-feature RMSNorm statistics.
C/write: retention(.99,.99,.99,.99),write(.20,.10,.05,.02),
four independent96-feature RMSNorm statistics.

S=g*S_previous + i*tanh(W_V*x), output=sigmoid(W_Q*x)*RMSNorm(S).
The declared residual wrapper adds x exactly once. Head-wise norm groups outputs;
all head-specific projection rows see the full384-feature input.
Gate weights start zero and biases are logits of declared initial probabilities;
all gates become input-dependent through training. FP32 bias/sigmoid and scan
accumulation under BF16. Original learned per-feature gains, epsilon1e-6 retained.
Original affine scan unchanged; no loop over heads/tokens, no new projections,
routing, attention, recurrence changes or output mixing. Global RMSNorm in A
couples readout normalization statistics, not the recurrent updates.
Constant384 FP32 recurrent values/sample/layer,1.5KiB; empty key cache.
Global/head cache IDs distinguish normalization semantics. Parameter layout
unchanged; run manifests identify configuration/checkpoint compatibility.

Same seed42,8192 tokenizer, pinned TinyStories30k train/256validation documents,
context256,batch8,width384,sixlayers,SwiGLU1024,BF16,AdamW3e-4,300warmup,
cosine schedule,clip1,9000updates/18429440actual targettokens,validation500.
Equal update/token budget; elapsed wall time is measured, not constrained equal.
Independent fresh random initialization, same initial unchanged parameters,
same data ordering. No warm starts. Separate checkpoints/logs/manifests.
All13,774,080parameters, identical recurrent capacity to reference.

695 tests per branch passed; Ruff lint/format passed. Includes FP64 sequential
reference values/all input/parameter gradients for all initialization/norm modes,
multi-feature-head references, exact original-global RMSNorm equivalence,
identical recurrent states under changed norm, per-head isolation, causality,
gradcheck, residual once, cache full/chunk/single-token/save/load/reset/constant
memory, zero/long inputs, BF16 FP32-cache/gate precision and finite gradients,
parameter counts, initial biases/RNG and invalid configuration validation.
Benchmarks/probes discard weights before independent training.

Reference comparisons:
- Standard global norm + uniform biases: loss2.090822,PPL8.091566.
- Uniform heads: loss2.092968,PPL8.108945.
- Specialized heads: loss2.023171,PPL7.562270.
- Lantern attention: loss2.018320,PPL7.525675 (12,440,064parameters).
A versus specialized heads isolates normalization under heterogeneous biases.
B/C versus uniform/specialized heads separate retention/write initializations.
B/C/fullyspecialized/uniform form a two-factor headnorm initialization ablation.
Single seed; no semantic specialization, retrieval or broad superiority claims.
