# Tide — continuously ranged initialization

Codename: **Tide**. Branch: `model/001-spoony-tide`. Frozen training artifacts retain their original `fluxv-init-range` names and training-time branch identity for reproducibility.

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

## Completed controlled comparison

Same seed 42, data/tokenizer fingerprints, optimizer/schedule settings, and 9,000 updates / 18,429,440 actual target tokens. All No-K models have13,774,080parameters,global RMSNorm and384FP32 recurrent values/sample/layer. Timing includes full fresh training/validation/saving, excluding benchmarks/probes; historical references were timed separately. Lantern has12,440,064parameters and a growing/windowed KV cache.

| Model | Loss | PPL | Minutes | Tokens/s | Peak GiB | Loss vs discrete |
|---|---:|---:|---:|---:|---:|---:|
| Standard .99/.5 | 2.090822 | 8.091566 | 6.28 | 48876 | 0.999 | +0.081679 |
| Discrete specialized/global | 2.009144 | 7.456929 | 6.30 | 48785 | 0.999 | +0.000000 |
| 1 Uniform mean | 2.027631 | 7.596071 | 6.29 | 48798 | 0.999 | +0.018488 |
| 2 Continuous range | 2.010153 | 7.464462 | 6.32 | 48630 | 0.999 | +0.001010 |
| 3 Uniform minimum | 2.022821 | 7.559623 | 6.32 | 48572 | 0.999 | +0.013678 |
| Lantern attention | 2.018320 | 7.525675 | 7.19 | 42727 | 0.748 | +0.009177 |

### Gate activity and stability

Four diagnostic bins are contiguous96-feature slices under global RMSNorm, not independently normalized memories. Statistics average per-layer held-out fixed-batch measurements across six layers. Min/max are averages of layer extrema; per-layer distributions remain in JSON. Token std averages per-feature variation across tokens. Output activity does not prove causal usefulness.

#### 1 Uniform mean, initial

Gradient norm 3.237156; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.95625 | 0.00000 | 0.95625 | 0.95625 | 0.09250 | 0.00000 | 0.09250 | 0.09250 | 0.50837 |
| 2 | 0.95625 | 0.00000 | 0.95625 | 0.95625 | 0.09250 | 0.00000 | 0.09250 | 0.09250 | 0.51677 |
| 3 | 0.95625 | 0.00000 | 0.95625 | 0.95625 | 0.09250 | 0.00000 | 0.09250 | 0.09250 | 0.52573 |
| 4 | 0.95625 | 0.00000 | 0.95625 | 0.95625 | 0.09250 | 0.00000 | 0.09250 | 0.09250 | 0.51724 |

#### 1 Uniform mean, final

Gradient norm 1.763425; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.85231 | 0.13603 | 0.03427 | 0.99910 | 0.09973 | 0.07324 | 0.00330 | 0.96117 | 0.50693 |
| 2 | 0.84915 | 0.14140 | 0.01178 | 0.99867 | 0.10231 | 0.07555 | 0.00397 | 0.93244 | 0.53932 |
| 3 | 0.84805 | 0.14046 | 0.00503 | 0.99950 | 0.10274 | 0.07620 | 0.00337 | 0.94772 | 0.52067 |
| 4 | 0.84861 | 0.14291 | 0.00446 | 0.99951 | 0.10512 | 0.07683 | 0.00314 | 0.97214 | 0.55217 |

#### 2 Continuous range, initial

Gradient norm 3.153794; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.91178 | 0.00000 | 0.90000 | 0.92356 | 0.17768 | 0.00000 | 0.15535 | 0.20000 | 0.55526 |
| 2 | 0.93559 | 0.00000 | 0.92381 | 0.94738 | 0.13256 | 0.00000 | 0.11023 | 0.15488 | 0.53525 |
| 3 | 0.95941 | 0.00000 | 0.94762 | 0.97119 | 0.08744 | 0.00000 | 0.06512 | 0.10977 | 0.51318 |
| 4 | 0.98322 | 0.00000 | 0.97144 | 0.99500 | 0.04232 | 0.00000 | 0.02000 | 0.06465 | 0.45810 |

#### 2 Continuous range, final

Gradient norm 1.724368; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.75408 | 0.19455 | 0.00294 | 0.99917 | 0.20303 | 0.12588 | 0.00763 | 0.96183 | 0.66457 |
| 2 | 0.80885 | 0.16238 | 0.01082 | 0.99855 | 0.14478 | 0.09552 | 0.00581 | 0.95732 | 0.56772 |
| 3 | 0.87384 | 0.12011 | 0.01393 | 0.99885 | 0.08767 | 0.06287 | 0.00407 | 0.93574 | 0.45541 |
| 4 | 0.93487 | 0.09087 | 0.06855 | 0.99964 | 0.04031 | 0.03509 | 0.00129 | 0.86826 | 0.42953 |

#### 3 Uniform minimum, initial

Gradient norm 2.854077; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.51086 |
| 2 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.51677 |
| 3 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.52392 |
| 4 | 0.90000 | 0.00000 | 0.90000 | 0.90000 | 0.02000 | 0.00000 | 0.02000 | 0.02000 | 0.51668 |

#### 3 Uniform minimum, final

Gradient norm 1.639613; all sampled gradients/states/outputs finite.

| Bin | g mean | g token std | g min | g max | i mean | i token std | i min | i max | Output RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.80164 | 0.15007 | 0.02141 | 0.99896 | 0.02506 | 0.02112 | 0.00104 | 0.70362 | 0.55118 |
| 2 | 0.79485 | 0.15612 | 0.00653 | 0.99898 | 0.02564 | 0.02159 | 0.00130 | 0.80298 | 0.56830 |
| 3 | 0.79237 | 0.15560 | 0.00639 | 0.99849 | 0.02608 | 0.02183 | 0.00136 | 0.81119 | 0.53728 |
| 4 | 0.79371 | 0.15946 | 0.00695 | 0.99911 | 0.02643 | 0.02200 | 0.00135 | 0.78627 | 0.56616 |

### Fixed-prompt generations

Same prompts,64new-token maximum,best saved checkpoint. Primary metrics use final-budget weights. These are story samples, not QA/retrieval tests.

#### 1 Uniform mean

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running towards her. Lily was scared and didn't know what to do.
> Suddenly, a big dog came running towards her. Lily was scared and ran away. The dog ran away and Lily

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not friendly. It barked and wagged its tail. Tom and Tom were happy. They played with the dog and the dog. They had fun and forgot about the dog. They were happy.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy, but he couldn't find it. He looked everywhere, but he couldn't find it.
> Then, he saw a big, red ball. He picked it up and took it home. He played with it all day. He was very happy. The little

#### 2 Continuous range

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was very scary and Lily was scared.
> Lily's mom told her to be careful and not to go near the dog. The dog was scared and didn't want to go. Lily was scared, but she

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was brown and had a long tail. Tom was very happy. He wanted to play with the dog.
> "Hi, dog," Tom said. "Can I pet you?"
> The dog looked at Tom and wagged his tail. He was not scared of the dog. He was not scared of the dog. He was

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to make the boy happy.
> He saw a boy who was sad and lonely. He had no friends to play with. He had no friends. He had no friends. He had no friends. He had no friends.
> One day, the boy saw a big dog. The boy

#### 3 Uniform minimum

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She wanted to play with it, but it was too high up.
> Lily asked her mom, "Can I play with the ball, please?" Her mom said, "Yes, but be careful."

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was not scary, but Tom was brave. He wanted to play with the dog.
> Tom and the dog played together for a while. They ran and ran, and the dog ran around the yard. They laughed and had fun. Tom was not scared anymore. He was happy to have a new friend.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to help the boy find his toy. He looked everywhere, but he could not find it.
> Then, he saw a big dog. The dog was friendly and wanted to play with the dog. The dog was friendly and wanted to play with the dog. The dog was happy and they

### Interpretation

Discrete specialized-global remains the lowest observed loss:2.009144.
Continuous range essentially matches it at2.010153 (+0.001010 loss,
+0.10% perplexity), without outperforming it. Four discrete initialization
groups therefore were not necessary to obtain a similar result in this seed.
The ranged recipe also changes mean probabilities, so the result does not
isolate smoothness alone.

Uniform mean gives2.027631 (+0.018488 loss,+1.87% perplexity versus discrete).
Matching the arithmetic means of g and i recovers approximately77% of the
loss improvement from standard .99/.5 initialization to discrete specialization.
Uniform minimum gives2.022821 (+0.013678 loss,+1.38% perplexity versus discrete),
recovering approximately83% of that loss improvement. Those percentages are
descriptive fractions on the loss scale, not causal contribution estimates.
Much of the observed gain is therefore achievable without heterogeneous
initial probabilities, but neither uniform recipe reproduces the full result.

The additional advantage of the varied profiles is consistent with a useful
initial distribution of gate preferences beyond probability means. It does
not establish specialized semantic memories or long-context retrieval.
These controls do not separately identify the effects of spread, long-retention
tails, retention/write anticorrelation or optimization dynamics. Mean matching
in probability space also does not match mean logits, gate gradients or
nonlinear memory timescales. Repeated seeds are needed to judge robustness,
particularly the very small discrete/range difference.

Final layer-averaged retention/write means by diagnostic feature quartile:
- Mean: g[0.8523,0.8491,0.8481,0.8486],i[0.0997,0.1023,0.1027,0.1051].
- Range: g[0.7541,0.8089,0.8738,0.9349],i[0.2030,0.1448,0.0877,0.0403].
- Minimum: g[0.8016,0.7949,0.7924,0.7937],i[0.0251,0.0256,0.0261,0.0264].

All bins/layers developed input-dependent gate variation and nonzero readout
activity. Similar quartile means in the uniform models do not imply identical
feature behavior: the parameters are independent and aggregation can hide
feature differences. The range model retains ordered aggregate tendencies;
these are not measured memory lifetimes or evidence of semantics.

Global RMSNorm largely cancels a uniform scalar write amplitude initially,
apart from epsilon. Low initial i therefore does not directly mean a similarly
small residual readout; it can still change raw state scale and optimization
of later input-dependent gates. Minimum changes g and i together, so it cannot
attribute its gain to suppressed writes alone.

All three runs finished9000updates/18429440tokens. Source/config/artifact hashes,
common initial weights, forward equivalence and settings audits passed.
Initial/final gradients and sampled activations were finite; no numerical
failure appeared in the saved validation curves. Large datasets/checkpoints
remain local and ignored. Prior branches and the user's standard are preserved.
