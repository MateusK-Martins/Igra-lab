# Specialized No-K mechanism ablations — write

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

## Completed controlled comparison

All runs: same seed42, data/tokenizer hashes, optimizer/schedule/settings,9000updates/18429440actualtargettokens. Wall time includes full fresh training, validation and checkpoint saving; excludes benchmark/probes. Historical controls were timed separately. No-K models all have384 FP32 state values/sample/layer; Lantern uses a growing/windowed KV cache and12,440,064parameters.

| Model | Parameters | Loss | Perplexity | Minutes | Tokens/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 13,774,080 | 2.090822 | 8.091566 | 6.28 | 48876 | 0.999 |
| Uniform heads | 13,774,080 | 2.092968 | 8.108945 | 6.30 | 48742 | 0.999 |
| Specialized heads | 13,774,080 | 2.023171 | 7.562270 | 6.33 | 48560 | 0.999 |
| A Specialized/global | 13,774,080 | 2.009144 | 7.456929 | 6.30 | 48785 | 0.999 |
| B Retention only | 13,774,080 | 2.051624 | 7.780530 | 6.33 | 48520 | 0.999 |
| C Write only | 13,774,080 | 2.051832 | 7.782146 | 6.34 | 48465 | 0.999 |
| Lantern attention | 12,440,064 | 2.018320 | 7.525675 | 7.19 | 42727 | 0.748 |

### Controlled effects on validation loss

Negative differences mean reduced loss. These are descriptive single-seed effects on the loss scale; no significance/robustness inference. Interaction is joint loss minus the sum of separate effect predictions.

| Contrast | Loss difference |
|---|---:|
| headnorm uniform minus global uniform | +0.002145 |
| headnorm specialized minus global specialized | +0.014028 |
| joint specialization global | -0.081679 |
| joint specialization headnorm | -0.069796 |
| retention specialization only headnorm | -0.041343 |
| write specialization only headnorm | -0.041136 |
| write added to retention headnorm | -0.028453 |
| retention added to write headnorm | -0.028661 |
| retention write loss interaction | +0.012682 |
| norm jointspecialization loss interaction | +0.011882 |

### Per-head gate distributions

Values average layer-wise held-out fixed-batch statistics over six layers. Min/max columns are means of per-layer extrema, not global extrema. Full per-layer ranges are in initial/final-probe.json. Token std is per-feature token variation, then averaged across batch/features. Output activity is not a test of causal usefulness or semantics. State/normalized RMS sample the final recurrent state; output RMS uses the full probe sequence.

#### A Specialized/global, initial

Gradient norm 3.154238; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.20000 | 0.00000 | 0.20000 | 0.20000 | 0.60262 |
| 2 | 0.95000 | 0.00000 | 0.95000 | 0.95000 | 0.10000 | 0.00000 | 0.10000 | 0.10000 | 0.51834 |
| 3 | 0.98000 | 0.00000 | 0.98000 | 0.98000 | 0.05000 | 0.00000 | 0.05000 | 0.05000 | 0.51402 |
| 4 | 0.99500 | 0.00000 | 0.99500 | 0.99500 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.41393 |

#### A Specialized/global, final

Gradient norm 1.645361; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.73458 | 0.20785 | 0.00204 | 0.99947 | 0.22888 | 0.13653 | 0.00766 | 0.96712 | 0.77141 |
| 2 | 0.84470 | 0.14247 | 0.01413 | 0.99870 | 0.10775 | 0.07686 | 0.00451 | 0.91859 | 0.54287 |
| 3 | 0.91936 | 0.09855 | 0.06240 | 0.99932 | 0.04824 | 0.03978 | 0.00246 | 0.87728 | 0.40725 |
| 4 | 0.95732 | 0.08172 | 0.04270 | 0.99983 | 0.01957 | 0.01905 | 0.00087 | 0.74809 | 0.33144 |

#### B Retention only, initial

Gradient norm 3.177023; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.50000 | 0.00000 | 0.50000 | 0.50000 | 0.51603 |
| 2 | 0.95000 | 0.00000 | 0.95000 | 0.95000 | 0.50000 | 0.00000 | 0.50000 | 0.50000 | 0.51628 |
| 3 | 0.98000 | 0.00000 | 0.98000 | 0.98000 | 0.50000 | 0.00000 | 0.50000 | 0.50000 | 0.51965 |
| 4 | 0.99500 | 0.00000 | 0.99500 | 0.99500 | 0.50000 | 0.00000 | 0.50000 | 0.50000 | 0.51640 |

#### B Retention only, final

Gradient norm 1.724351; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.70870 | 0.22620 | 0.00161 | 0.99942 | 0.38668 | 0.19904 | 0.01077 | 0.98518 | 0.71823 |
| 2 | 0.87226 | 0.13120 | 0.01685 | 0.99896 | 0.42928 | 0.18883 | 0.01621 | 0.98643 | 0.54243 |
| 3 | 0.94677 | 0.08126 | 0.02940 | 0.99951 | 0.46480 | 0.18023 | 0.02359 | 0.98718 | 0.44202 |
| 4 | 0.97734 | 0.04690 | 0.01383 | 0.99988 | 0.46631 | 0.17397 | 0.02225 | 0.98428 | 0.42065 |

#### C Write only, initial

Gradient norm 3.216714; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99000 | 0.00000 | 0.99000 | 0.99000 | 0.20000 | 0.00000 | 0.20000 | 0.20000 | 0.51537 |
| 2 | 0.99000 | 0.00000 | 0.99000 | 0.99000 | 0.10000 | 0.00000 | 0.10000 | 0.10000 | 0.51554 |
| 3 | 0.99000 | 0.00000 | 0.99000 | 0.99000 | 0.05000 | 0.00000 | 0.05000 | 0.05000 | 0.51969 |
| 4 | 0.99000 | 0.00000 | 0.99000 | 0.99000 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.51581 |

#### C Write only, final

Gradient norm 1.752615; all sampled gradients/activations finite.

| Head | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.93775 | 0.08309 | 0.07381 | 0.99964 | 0.21100 | 0.11666 | 0.00859 | 0.95953 | 0.44368 |
| 2 | 0.90251 | 0.12066 | 0.03117 | 0.99967 | 0.11558 | 0.08133 | 0.00445 | 0.96201 | 0.47410 |
| 3 | 0.91538 | 0.11205 | 0.03022 | 0.99971 | 0.05966 | 0.04906 | 0.00169 | 0.93573 | 0.45270 |
| 4 | 0.89351 | 0.13578 | 0.01568 | 0.99981 | 0.02691 | 0.02752 | 0.00088 | 0.88319 | 0.54404 |

### Fixed-prompt generations

Same prompts,64new-token maximum, best saved checkpoint. Primary metrics use final-budget weights; generations are illustrative and not QA/retrieval tests.

#### A Specialized/global

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running around. Lily was scared of the dog.
> But then, the dog saw her and ran away. Lily was very scared and didn't know what to do. She ran to her mommy

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and wanted to play with Tom. He ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog wagged his tail and wagged his tail.
> Tom and the dog played with the dog all day. They ran and jumped and laughed. They were very happy.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad for it and wanted to help it. He looked everywhere for it but couldn't find it.
> The boy asked his mom for help. She said, "Mom, can you help me find my toy?" His mom said, "Yes, I can help you."
> They looked everywhere for the toy, but

#### B Retention only

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very big and had sharp teeth. Lily was scared of the dog, but she was brave.
> She ran to the dog and said, "Hello, dog! Do you want to play with me?"

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was brown and had a long tail. The dog was not scared of the dog. He wanted to play with the dog.
> "Can I play with you?" Tom asked the dog. The dog barked and wagged his tail.
> "No, I want to play with you," Tom said. "I want to play

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to find a friend. He looked everywhere for the toy. He looked under the bed, in the closet, and in the closet, but he could not find it. He felt sad and lonely.
> Then, he saw a big dog. The dog was not friendly. It barked and

#### C Write only

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very friendly and Lily wanted to play with him.
> Lily ran to the dog and said, "Hi, dog! Do you want to play with me?" The dog wagged its tail and wagged its tail

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not friendly. He wanted to play with Tom and the dog.
> Tom and the dog played with the ball. They ran and ran and jumped. They laughed and had fun. Tom was happy. He was not scared anymore. He was happy. He had a new friend.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad for the boy and wanted to help him. He went to his mom and said, "Mom, can you help me find my toy?"
> His mom smiled and said, "Yes, I can help you. But you have to be careful and not touch anything. It's not safe."
> The boy looked

### Interpretation

A is the best observed result: loss2.009144 versus2.023171 for specialized
head-wise normalization, and2.090822 for standard No-K. Its perplexity is
7.84% lower than standard. Independent head normalization is therefore not
necessary for this initialization recipe to help on this seed; global
normalization performed better. With global normalization the four heads
are just labels for feature groups with different initial biases. A has the
same computational architecture and parameter count as standard No-K:
this gain can be obtained through initialization without a new mechanism.

Under head-wise normalization, retention-only and write-only improvements
are nearly identical: -0.041343 and-0.041136 loss versus uniform heads.
Each recovers about59% of the joint specialization loss improvement.
Neither alone matches joint specialization. Adding write specialization to
retention-only improves loss0.028453; adding retention specialization to
write-only improves loss0.028661.

There is no super-additive benefit on the loss scale. Adding the individual
loss reductions predicts2.010489, whereas joint head-wise specialization
actually gives2.023171. The interaction is+0.012682, consistent with
overlapping/diminishing gains in this one run. It is descriptive, not a
statistical interaction estimate across seeds.

A's layer-averaged final retention means are[0.7346,0.8447,0.9194,0.9573],
and write means[0.2289,0.1078,0.0482,0.0196]. Retention-only also retains
ordered mean forgetting preferences; write-only retains ordered writing
preferences but has no ordered retention sequence. Every head in every
tested layer has nonzero token variation in both learned gates. All have
nonzero output activity; these observations do not establish necessity,
semantic specialization, retrieval quality or measured memory lifetimes.

These are initialization/normalization ablations, not changes to state
capacity. The heterogeneous recipes change mean initial gate probability
as well as diversity; mean-matched uniform controls were not run. Thus
the results cannot establish that diversity itself, rather than shorter
initial retention, lower write biases or optimization effects, caused the
benefit. Under head-wise RMSNorm, a constant per-head write scale largely
cancels at initialization except for epsilon, while gate gradients and
subsequent input-dependent writing can still differ.

A's loss is0.009177 below historical Lantern, but No-K has13,774,080 versus
Lantern's12,440,064 parameters, a different architecture/cache and different
initialization. This single-seed result is not broad architecture superiority.
Full curves and fixed generations remain part of the comparison; low loss
does not establish QA, reasoning or long-context retrieval.

Source/config fingerprints stayed frozen throughout all three runs; data,
tokenizer, optimizer/schedule settings, target exposure and unchanged initial
parameters were audited. Initial/final gradient probes were finite. Training
time and peak VRAM stayed near standard. The standard model remains unchanged;
none of these experiments was automatically promoted.
