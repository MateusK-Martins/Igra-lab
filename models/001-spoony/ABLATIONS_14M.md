# Controlled FluxV 14M ablations

Reference: 29890bc8a7327667102eceb823317a009b96d32e. Historical V-only result:
13,774,080 parameters, loss 2.1738922596, PPL 8.792440, 9,000 updates,
18,429,440 actual target tokens (partial epoch-ending batches explain the small
shortfall from nominal 18,432,000). Historical artifacts are preserved.

Every branch starts fresh from seed 42 with the original tokenizer and exact
train/validation stores (30,000 / 256 TinyStories documents). Dataset fingerprints
and all resolved training settings are checked against the historical manifest.
Context 256, batch 8, BF16, AdamW LR 3e-4, betas(.9,.95), wd.01, warmup 300, cosine
floor.1, clip 1, accumulation 1, validation 500, checkpoint 1000, 9,000 updates.
The existing training, optimizer, loader, scan and generation scripts are reused.
No QA stage is part of this experiment. No original branch/artifact is modified.

| Model | K/V normalization | Readout | Parameters |
| --- | --- | --- | ---: |
| Reference | Global RMSNorm | Element-wise sigmoid Q times V | 13,774,080 |
| A | Six independent 64-feature RMS statistics | Original element-wise | 13,774,080 |
| B | Same head normalization as A | Cyclic factored matrix interactions | 13,774,080 |
| C | Original global RMSNorm | Identical matrix interactions | 13,774,080 |
| D | Original global RMSNorm | Residual shared head-axis 6→24→6 GELU MLP | 13,775,988 |
| E | No heads; RMS-normalized 128-feature matrix readout | Single decayed 64×128 matrix | 14,211,456 |

A–D retain full-width cumulative K, token-dependent V decay, original FP32 vector
states, existing affine scan, outer norms, FFN, embedding and residual wrappers.
Head normalization keeps 384 learned gains for each norm and epsilon 1e-6; only
normalization domains change. Matrix B/C uses q=sigmoid(Wq x), with 288 outputs,
P:[6,64,48], W:[6,48,64], u_h=v_h P_h, z_h=v_(h+1 mod6)*((q_h*u_h)W_h),
concatenated in cyclic-pair order. No explicit temporary matrices are allocated.
P and W are initialized independently Normal(0,1/sqrt(fan_in)); fixed output
scaling is 1. Reduced Q saves 36,864 parameters/layer, exactly offset by P+W.

Initialization draws the original full Q before replacing it inside a CPU RNG
fork. Extra P/W and MLP initialization also uses forks, preserving the random
stream and unchanged parameter initial values. B and C start with identical
weights. D's MLP includes biases (318 parameters/layer; 1,908 total); final
projection weight/bias are zero, making its initial readout exactly the baseline.
No trainable extra gains, routing, output projections or head semantics claims.

E is a separate architecture, not part of the 2×2 normalization/readout ablation.
It retains full-width bias-free Q/K/V, applies sigmoid/tanh/sigmoid respectively,
then bias-free down-projections to 64/64/128. Its write is
(k_down(tanh(Kx))/sqrt64) outer v_down(sigmoid(Vx)). Gate row/column projections
384→64 and 384→128 are FP32 sigmoid; zero weights, biases logit(sqrt(.99)). Their
outer product gates the previous matrix element-wise, before the current write.
Q reads the updated matrix with an additional fixed 1/sqrt64 query scale. The
128-dimensional read gets trainable global RMSNorm(eps1e-6), then bias-free128→384.
There is no separate cumulative K or V. Outer FFN/norms are unchanged. Count is
437,376 above baseline (+3.175%); no artificial capacity padding.

E uses the original affine_prefix_sum on flattened matrix states inside chunks
of 16 tokens, with non-reentrant PyTorch activation checkpointing. Chunk boundaries
are sequential, gradients cross them without detach. Within each chunk, the scan
remains parallel. This changes execution cost, not recurrence/context/exposure;
no scan optimization is attempted. Training materializes only local [B,16,64,128]
matrices; inference stores a single [B,64,128] FP32 matrix per layer in its own
MatrixStateEntry. Vector caches cannot be reinterpreted as matrix caches. The
matrix uses 8192 state elements/sample/layer vs 768 for two vectors (10.67×).

Validation curves and fixed greedy prompts use the original procedures. Reports
show final-budget loss/PPL as primary; inherited run_baseline also records the best
saved checkpoint and generations from that checkpoint, explicitly labeled.
Initial/final fixed-held-out-batch probes measure mixer output RMS/variance,
per-head state/output amplitudes for A–D, gate distributions/token variation,
P/W norms, D MLP relative change, finite gradients and total gradient norm. E
records matrix norm, sample 0 singular values/entropy effective rank and query
sensitivity on fixed memory. These diagnostics are not reasoning/retrieval tests.
Training itself retains the existing finite-gradient check at every BF16 update.

Commands: python -m scripts.probe_variants initial; python -m scripts.benchmark;
python -m scripts.run_baseline; python -m scripts.probe_variants final. Use the
model .venv; benchmarking/probes are separate processes so they never alter fresh
training initialization or loader RNG. Data and checkpoints remain ignored.

Preflight complete: A–D full suites 636 tests each; E 623 tests, including full-size
CUDA memory/backward. Ruff passes. Every data/settings audit matches the historical
manifest. Historical activation/gradient probes are reproduced from a temporary
checkout of the exact reference commit, loading its preserved final weights only
for its final probe. Benchmark and probe weights are always discarded before fresh
training. E also forks extra-projection initialization to preserve unchanged outer
FFN and later-layer seed values.

## Completed comparison

All five fresh experiments reached 9,000 updates and exactly 18,429,440 target tokens.
Data/tokenizer hashes, seed, batch order and complete training settings match the
historical V-only baseline. Source fingerprints stayed frozen. All 18 validation
measurements are preserved. A resumed after a suspend-related GPU failure; E
resumed after a user-requested stop. No architecture or budget was changed.

| Model | Parameters | Final loss | PPL | Measured phase min | Phase updates | Phase tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 9,000 | 46,589 | 1.044 |
| A | 13,774,080 | 2.177169 | 8.8213 | 4.48 | 6,000 | 45,682 | 1.045 |
| B | 13,774,080 | 2.190446 | 8.9392 | 6.85 | 9,000 | 44,824 | 1.081 |
| C | 13,774,080 | 2.187075 | 8.9091 | 6.77 | 9,000 | 45,364 | 1.080 |
| D | 13,775,988 | 2.171528 | 8.7717 | 7.07 | 9,000 | 43,472 | 1.130 |
| E | 14,211,456 | 2.039836 | 7.6893 | 35.04 | 8,000 | 7,792 | 1.044 |

A/E timing covers resumed phases only, not all 9,000 updates; their throughput
denominators likewise cover only newly trained phase tokens. Do not compare these
phase durations to complete-run durations. Fresh 100-update benchmark timings are
also included for all models. Peak CUDA for A/E is measured during resumed phases.

### Controlled 2×2 effects

Positive loss differences are worse.

- A_minus_original: +0.003276 nats/token.
- B_minus_C: +0.003371 nats/token.
- C_minus_original: +0.013183 nats/token.
- B_minus_A: +0.013278 nats/token.
- interaction_B_minus_A_minus_C_plus_original: +0.000095 nats/token.
- D_minus_original: -0.002365 nats/token.
- E_minus_original: -0.134057 nats/token.

A did not improve the baseline. Matrix readout B/C slightly worsened loss under
both normalization choices. B−C and A−baseline are similar; the interaction
contrast is close to zero in this seed, giving no strong observed interaction.
D's tiny change should not be interpreted as an established advantage. E lowered
loss by 0.134057 nats/token (perplexity 8.7924 → 7.6893) in this run. E is a
separate memory experiment with 3.18% extra parameters, 10.67× recurrent state
elements, and higher computational cost. Its result cannot isolate memory type
from those changes, and perplexity does not establish retrieval or reasoning.

### Diagnostic evidence

Initial/final probes record mixer activation RMS, finite gradients and learned
token-decay distributions. A–D report per-head normalized-state/output RMS,
variance and magnitude. B/C also report learned P/W norms. Nonzero amplitudes
or different decay rates do not establish different head semantics or complementary
information. Inspect the full per-layer JSON rather than infer specialization.

D starts exactly at identity. On the fixed final CPU FP32 state probe, MLP RMS
changes range 7.5–28.8% across layers. Analytic off-diagonal Jacobians demonstrate
learned cross-head coupling; off-diagonal energy is only 0.21–1.59% of full mapping
Jacobian energy, so the identity path still dominates. The analytic probe uses
J_delta=W2 diag(Phi(z)+z phi(z)) W1 and reports its full per-layer matrices.

E records matrix norms, all 64 singular values and entropy effective rank on
sample 0 of the first held-out batch. Query sensitivity compares the current final
query with its feature-reversed version against the same cached final matrix.
These are stability/representation diagnostics, not task-level retrieval tests.
The matrix cache holds one [B,64,128] FP32 tensor per layer, never a token-growing
sequence. Activation checkpointing retains chunk-boundary matrices for backpropagation;
local parallel scans and checkpoint recomputation add execution overhead.

Fixed-prompt generations and full curves are in training-report.json and the shared
comparison.json. Each branch keeps its own full initial/final probes, initializer
audit, benchmark, baseline audit and manifest. Checkpoints/data remain local and
ignored. Tests 636 each A–D and 623 E, plus Ruff, passed before expensive training.
The existing finite-gradient guard ran at every completed BF16 optimizer step.

### Measured diagnostic summaries

RMS and decay means below are averaged over the six layers of the fixed held-out
batch probe. E's matrix gate is sampled on the first 16 tokens; vector-gate
statistics use the full 256-token batch. Gradient norms are un-clipped diagnostic
batch norms, rather than the training clipping threshold.

| Model | Mixer output RMS, initial → final | Gradient norm, initial → final | Final retention mean | Retention token variation std |
| --- | --- | --- | ---: | ---: |
| Original | 0.5176 → 0.6730 | 2.5160 → 3.2867 | 0.84438 | 0.16740 |
| A | 0.5178 → 0.6542 | 2.5306 → 2.0388 | 0.85861 | 0.15687 |
| B | 0.5160 → 0.6830 | 3.8545 → 3.0069 | 0.89414 | 0.13062 |
| C | 0.5160 → 0.6624 | 3.7863 → 2.8086 | 0.87475 | 0.14933 |
| D | 0.5176 → 0.7080 | 2.5183 → 2.7028 | 0.84901 | 0.16545 |
| E | 0.5548 → 0.4958 | 4.0980 → 3.3342 | 0.86280 | 0.15292 |

All initial/final diagnostic gradients and outputs are finite. Initial retention
is approximately 0.99 for every model; nonzero final token variation shows learned
input dependence on the sampled batch. It does not establish semantic roles.

Per-head amplitude ratios (smallest/largest output RMS within a layer), across
sampled layers, are descriptive utilization checks, not measures of independence:

- Original: ratio range 0.6223–0.8363. Full per-head RMS, variance and magnitude remain in the JSON probes.
- A: ratio range 0.4495–0.9217. Full per-head RMS, variance and magnitude remain in the JSON probes.
- B: ratio range 0.1790–0.7341. Full per-head RMS, variance and magnitude remain in the JSON probes.
- C: ratio range 0.3887–0.7020. Full per-head RMS, variance and magnitude remain in the JSON probes.
- D: ratio range 0.5889–0.8140. Full per-head RMS, variance and magnitude remain in the JSON probes.
- B learned P norms: 6.6040–7.1931; W norms: 7.6731–8.2676. Initial per-pair values are in initial-probe.json.
- C learned P norms: 6.6086–7.1836; W norms: 7.7541–8.3188. Initial per-pair values are in initial-probe.json.

E's final matrix diagnostics (sample 0 effective rank; norms averaged over batch):

| Layer | Mean matrix Frobenius norm | Entropy effective rank | Query readout RMS difference |
| --- | ---: | ---: | ---: |
| body.1.branch | 19.6711 | 13.9262 | 0.57893 |
| body.5.branch | 12.6907 | 11.5260 | 0.29217 |
| body.9.branch | 308.7759 | 3.0558 | 18.87301 |
| body.13.branch | 12.8471 | 21.0260 | 0.20045 |
| body.17.branch | 97.3682 | 7.2421 | 2.92310 |
| body.21.branch | 397.0285 | 2.4521 | 23.79154 |

Effective rank is exp(−sum(p log p)), with p equal to normalized singular values,
not squared singular values. All 64 singular values per sampled layer are saved.
The query comparison holds the matrix fixed and reverses query feature order.
It shows a responsive readout, not demonstrated retrieval accuracy. Effective
rank varies from 2.45 to 21.03 out of 64 possible singular values; some layers
are dominated by a few singular directions. Matrix norm variation is substantial,
while normalized readouts and gradients remain finite on this probe.

### Fixed-prompt greedy generations

These retain the original generation procedure, using each best saved checkpoint.
Loss/PPL comparisons above remain based on the final-budget checkpoint.

#### Original — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was scared and ran away. Lily was scared and didn't know what to do.
> Suddenly, a big dog came running towards her. Lily was scared and ran away. She ran back to the dog and

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy. He wanted to play with the ball. He ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog looked at the ball and said, "Yes, I want to play with you."
> The dog and the dog played together. They played with the

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his toy car. He looked everywhere for it, but he could not find it. He looked everywhere for it, but he could not find it.
> Then, he saw a big dog. The dog was playing with a ball. The dog wanted to play with the ball

#### A — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big, scary dog. The dog was scared and ran away. Lily felt sad for the dog and wanted to help.
> Lily's mom saw the dog and asked, "What's wrong, Lily?" The dog barked and wagged its tail.

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not happy. It wanted to play with the dog.
> "Can we play with the dog?" Tom asked.
> "Yes, but be careful. The dog is not nice. It is not nice. It is not nice. It is not nice." Mia said.
> "I am sorry. I was mean

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his toy car, but he was too small to play with it. He looked everywhere, but he could not find it.
> One day, he saw a big dog in the park. The dog was not nice. The dog wanted to play with the dog. The dog

#### B — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very friendly and had a big smile on its face.
> Lily and the dog played together for a while, but then it started to rain. The dog ran away and Lily was sad. She wanted to

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy. He wanted to play with the ball. He ran to the ball and tried to catch it. But the ball was too fast. The ball went away.
> Tim was sad. He wanted to play with the ball. He ran to the ball and tried to catch it. But the ball was too high

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy car. He asked his mom for help. His mom said, "Yes, you can have it. But first, you have to be careful. It's not safe."
> Tim was sad. He wanted to play with his toy car. He said, "

#### C — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball. She wanted to catch it, but it was too high up.
> Lily's mom saw her and said, "Don't worry, Lily. We can fix it. We can fix it." They went to the

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very friendly and wanted to play with him. He ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog wagged its tail and wagged its tail.
> The dog wagged its tail and wagged its tail. It wagged its tail and wagged its tail. It wagged its tail

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy car. He looked everywhere for it, but he could not find it. He looked everywhere, but he could not find it.
> Then, he saw a big dog. The dog was playing with a ball. The dog was not nice. The dog was not

#### D — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very big and had a long tail. Lily wanted to pet the dog, but she was scared.
> Lily's mom saw her and said, "Don't worry, Lily. We can help you find

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy. He wanted to play with the ball. He ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog barked and wagged its tail.
> The dog barked and wagged its tail. It barked and wagged its tail. It barked and wagged its tail. It

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with his toy car. He looked everywhere for it, but he could not find it. He looked everywhere for it, but he could not find it.
> Then, he saw a big dog. The dog was playing with a ball. The dog wanted to play with the ball

#### E — checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very big and had a long tail. Lily was scared and didn't know what to do.
> Lily's mom saw her and said, "Don't worry, Lily. I will help you." She

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not happy. It wanted to play with Tom and Tom.
> "Let's play with the dog!" Tom said. He ran to the dog and tried to catch it. But the dog was faster. He pushed Tom and Tom away. Tom was sad. He did not want to play with Tom anymore.
> Tom

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to find his friend, the boy, to help him.
> The boy went to the boy and said, "Hi, I am Tom. I like to play with you. Do you want to play with me?" The boy said, "Yes, I want to play with you.
