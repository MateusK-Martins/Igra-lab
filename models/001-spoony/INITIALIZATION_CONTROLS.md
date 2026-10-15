# Specialized-global initialization controls — range

Architecture unchanged from specialized-global recipe05ad3d9, completed8985395:
signed No-K state S=g*S_previous+i*tanh(W_V*x), global learned RMSNorm,
output=sigmoid(W_Q*x)*RMSNorm(S), existing wrapper adds x once.
13,774,080 parameters;384 FP32 recurrent values/sample/layer,1.5KiB.
Width384,sixlayers,SwiGLU1024,vocab8192, same projections and parallel scan.
Only retention/write biases change, via logit(initial probability).
Both gate weights start zero and remain learnable, input-dependent after learning.
FP32 bias/sigmoid and accumulation under BF16. No separate head normalization.

1/mean: all features g0.95625,i0.0925 (arithmetic probability means of original).
2/range: u_j=j/383, j=0..383; g_j=.90+.095*u_j,i_j=.20-.18*u_j.
Endpoints inclusive; smooth opposing feature profiles; means g0.9475,i0.11.
3/minimum: all features g0.90,i0.02.
Same profile repeats independently in each of six layers. Four96-feature
diagnostic bins are just contiguous slices; under global norm they are not
independently normalized heads. Continuous profile varies within every bin.

Same frozen TinyStories30k train/256validation documents/tokenizer hashes,
seed42,context256,batch8,BF16,AdamW3e-4,300warmup,cosinefloor0.1,clip1,
9000updates/18429440actual targettokens,validation500/checkpoints1000.
Fresh independent training; shared initial parameters and loader order match.
Equal update/token exposure; elapsed duration measured, not fixed equal.

Minimal initialization-only support accepts one probability per feature in
addition to the existing per-group list. Production forward AST is identical.
700 tests passed per branch plus Ruff lint/format. Added exact profile/RNG/
unchanged-weight checks, independent FP64 reference values/all gradients,
BF16 FP32 gate precision/gradient stability, endpoints/means/monotonicity
and invalid lengths. Existing causal/cache/state/normalization tests retained.
Preflight verifies architecture/forward, count, profile, data/settings and
unchanged common initial parameters; benchmark/probes discarded before training.

Primary reference: four discrete groups with global norm, loss2.009144,
PPL7.456929. Mean versus reference controls arithmetic mean probabilities,
not mean logits or nonlinear effective time constants. Range differs from
reference in mean as well as distribution, so cannot isolate continuity alone.
Minimum changes retention/write tendencies together. Single seed; initialization
effects are not proof of semantic specialization, retrieval or general superiority.
Historical reports remain available; no automatic promotion of standard.
