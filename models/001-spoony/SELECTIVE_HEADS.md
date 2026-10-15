# Selective-write No-K heads: specialized

Standard reference is No-K + independent write gate + global RMSNorm (dcebd87,
recipe97f48aa), loss2.090822/PPL8.091566. Squared-retention model is not the standard.

Four96-feature heads, fixed384-element FP32 recurrent state per layer.
S_h = sigmoid(W_g_h*x+b_g_h)*S_h_prev + sigmoid(W_I_h*x+b_I_h)*tanh(W_V_h*x)
r_h = sigmoid(W_Q_h*x)*HeadRMSNorm(S_h)
Output is concatenated r_h; existing residual wrapper adds x exactly once.
Original full-width projections are row partitions (each head sees all384 inputs).
Retention/write decisions already independent featurewise in standard No-K.
New mechanism changes normalization domains; no new projection, attention,
routing, extra mixing, recurrent matrix, cross-head competition or token loop.
Original affine scan processes all384 features together unchanged.
Per-feature learned RMSNorm gains and epsilon1e-6 preserved, no gain sharing.
Parameter count13,774,080, identical to standard. Cache1.5KiB/sample/layer,
independent of sequence length. Four-head cache IDs distinguish global baseline.

Uniform A: all g_init0.99, i_init0.5, matching standard reference.
Specialized B: g_init(0.90,0.95,0.98,0.995), i_init(0.20,0.10,0.05,0.02).
Gate weights start zero; biases are logits of declared probabilities repeated96.
Bias addition/sigmoid and recurrent accumulation remain FP32 under BF16.
All gates learned from full current input after training. Only these declared
biases differ between A/B; common initial parameters and RNG streams match.
No claim that initial biases guarantee distinct semantic memories. RMSNorm
largely cancels uniform write scale within a head except for epsilon, so lower
initial write gate is not by itself evidence of lasting selectivity.

Same frozen TinyStories tokenizer/data30k train/256validation docs, seed42,
context256,batch8,BF16,FFN1024,width384,sixlayers,AdamW3e-4,warmup300,
9000updates/target18429440tokens,validation500,checkpoint1000.
Fresh independent runs; same loader seed/order. No budget or scan changes.

675 tests per branch passed, including independent FP64 sequential output and
parameter/input-gradient references with multi-feature heads, headnorm group
isolation, initialization/RNG, causality/gradcheck, residual once, full/chunk/
single-token caching/save/load/reset/constant memory, BF16 precision and finite
gradients, long/zero sequences, incompatible configurations and parameter count.
Ruff lint/format passed. Benchmarks/probes discarded before fresh training.

## Completed controlled comparison

All three have13,774,080 parameters, the same384 FP32 state values/sample/layer, same data/tokenizer hashes/settings, seed42 and9000updates/18429440tokens. Timing covers full fresh training plus validation/saving, excluding benchmarks/probes. Historical standard is separately timed.

| Model | Loss | Perplexity | Minutes | Tokens/s | Peak GiB |
|---|---:|---:|---:|---:|---:|
| Standard global RMSNorm | 2.090822 | 8.091566 | 6.28 | 48876 | 0.999 |
| A Uniform heads | 2.092968 | 8.108945 | 6.30 | 48742 | 0.999 |
| B Specialized heads | 2.023171 | 7.562270 | 6.33 | 48560 | 0.999 |

A versus standard isolates head-normalization grouping. B versus A tests the combined retention/write initialization bias; it cannot separate which bias causes a difference. B versus standard changes both normalization domains and initial gate preferences. All projections remain independent row groups viewing the entire input, as in the standard vector recurrence. This is a single-seed language-modeling comparison, not a reasoning/retrieval or semantic specialization test.

### Learned head activity

Aggregates below average layer-wise statistics over all six layers of the fixed held-out batch. Token std is averaged per-feature variation across tokens, not variation between heads. Output RMS/contribution measures do not establish causal usefulness.

#### A Uniform heads, initial

Gradient norm 3.156273; all gradients finite.

| Head | Retention mean | Retention token std | Write mean | Write token std | State RMS | Normalized RMS | Output RMS | Mean abs output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.99000 | 0.00000 | 0.50000 | 0.00000 | 11.20346 | 1.00000 | 0.51537 | 0.40631 |
| 2 | 0.99000 | 0.00000 | 0.50000 | 0.00000 | 11.58896 | 1.00000 | 0.51554 | 0.40654 |
| 3 | 0.99000 | 0.00000 | 0.50000 | 0.00000 | 11.74206 | 1.00000 | 0.51970 | 0.41242 |
| 4 | 0.99000 | 0.00000 | 0.50000 | 0.00000 | 11.68147 | 1.00000 | 0.51585 | 0.40607 |

#### A Uniform heads, final

Gradient norm 1.707640; all gradients finite.

| Head | Retention mean | Retention token std | Write mean | Write token std | State RMS | Normalized RMS | Output RMS | Mean abs output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.92334 | 0.11154 | 0.45938 | 0.17713 | 5.52130 | 0.98957 | 0.48114 | 0.33449 |
| 2 | 0.89320 | 0.14194 | 0.44439 | 0.17994 | 4.60993 | 1.00782 | 0.49638 | 0.33538 |
| 3 | 0.91738 | 0.11850 | 0.45613 | 0.17510 | 4.92603 | 0.98827 | 0.47604 | 0.32493 |
| 4 | 0.89974 | 0.13877 | 0.43409 | 0.17971 | 3.92306 | 1.00297 | 0.50685 | 0.34340 |

#### B Specialized heads, initial

Gradient norm 3.239111; all gradients finite.

| Head | Retention mean | Retention token std | Write mean | Write token std | State RMS | Normalized RMS | Output RMS | Mean abs output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90000 | 0.00000 | 0.20000 | 0.00000 | 0.52966 | 1.00000 | 0.51603 | 0.40720 |
| 2 | 0.95000 | 0.00000 | 0.10000 | 0.00000 | 0.51136 | 1.00000 | 0.51627 | 0.40678 |
| 3 | 0.98000 | 0.00000 | 0.05000 | 0.00000 | 0.61229 | 1.00000 | 0.51965 | 0.41160 |
| 4 | 0.99500 | 0.00000 | 0.02000 | 0.00000 | 0.65410 | 1.00000 | 0.51636 | 0.40575 |

#### B Specialized heads, final

Gradient norm 1.593623; all gradients finite.

| Head | Retention mean | Retention token std | Write mean | Write token std | State RMS | Normalized RMS | Output RMS | Mean abs output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.70989 | 0.21948 | 0.18902 | 0.12776 | 0.35983 | 1.07324 | 0.70254 | 0.47378 |
| 2 | 0.87565 | 0.12108 | 0.10659 | 0.08127 | 0.43350 | 1.00189 | 0.54660 | 0.35243 |
| 3 | 0.94904 | 0.07179 | 0.05817 | 0.04890 | 1.16993 | 0.97147 | 0.42787 | 0.25190 |
| 4 | 0.97727 | 0.04532 | 0.02490 | 0.02125 | 2.16864 | 0.94506 | 0.39123 | 0.20654 |

Per-layer gate ranges/query statistics and head-state cosine samples are retained in initial/final-probe.json. Cosines compare equal feature indices across independent heads in one sampled state batch; head coordinates are learned independently, so these are weak diagnostics, not semantic similarity evidence. Nonzero output magnitude alone does not prove useful specialization.

### Fixed-prompt greedy samples

Same prompts,64new-token maximum, best saved checkpoint. Primary metrics use final-budget weights.

#### A Uniform heads

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, shiny rock in the grass. She picked it up and showed it to her mom. "Look, Mommy! I found a big rock!" she said.
> Her mom smiled and said, "That's a nice rock, Lily.

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and thanked Tom. He said, "Thank you, dog. You are a good dog." Tom and the dog played together all day. They were best friends.

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy, but he could not find it. He looked everywhere, but he could not find it.
> Then, a kind girl named Lily came to the park. She saw the toy and said, "Hi, I am Lily. Do you want to play with me

#### B Specialized heads

Checkpoint: checkpoints/step-00009000.pt

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. She wanted to catch it, but it was too high up.
> Lily asked her friend, "Can you help me get the ball?" Her friend said, "Yes, I can help you." So

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was big and brown. He had a long tail and a long tail. He was very happy.
> "Look, Tom, a dog!" Tom said. "He is a dog. He is very friendly. He likes to play with us."
> "Can I pet him?" Tom asked.
> "Sure, but

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to make the boy happy.
> He looked around and saw a big tree. He thought, "I can make a big tree with my leaves." He picked up the tree and started to climb. He was very happy.
> But then, the tree started to rain. The tree was

### Interpretation

Uniform grouping alone did not improve this seed: loss increased0.00215
(+0.21% perplexity) versus standard global normalization. Specialized
initialization improved loss0.06980 versus uniform heads and0.06765 versus
standard (6.54% lower perplexity). Counts, state size and training exposure
match exactly; runtime and VRAM were nearly unchanged.

Specialized final mean retention remained ordered across heads:
[0.710,0.876,0.949,0.977], with mean writes[0.189,0.107,0.058,0.025].
These are layer-averaged held-out statistics, not fixed coefficients or
measured memory lifetimes. Gates became input-dependent in every head.
Uniform heads ended with much more similar retention/write means.
All specialized heads retained substantial normalized/output magnitude
(output RMS0.391–0.703); there is no obvious zero-output collapse on this
batch. This does not show that each head is necessary: controlled head
ablations or targeted memory tasks would be needed for that claim.

The observed gain is consistent with a useful bias toward different
writing/forgetting behaviors. It does not establish improved long-range
retention, retrieval or semantic specialization. Both retention and write
initialization changed together; a global-normalization model with the
same heterogeneous biases was not tested, so this cannot establish that
head-wise normalization is required for the gain. Distinct parameter
scales can also affect optimization. Repeat seeds are needed to establish
robustness. Fixed generations still show repetition and inconsistencies.

The user's standard remains the existing global-RMSNorm write-gated No-K;
these are separate experimental branches, with no automatic promotion.
