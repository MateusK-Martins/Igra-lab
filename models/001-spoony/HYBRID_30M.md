# Tide/Lantern hybrid 30M pipeline

[Tide,Tide,Lantern] repeated three times: six Tide and three Lantern mixers,
width512, nine layers, SwiGLU1152, tied8192 embedding/head initstd.02.
28,395,520parameters, 4.00% below Tide29,579,776. Preserve Tide's FFN rather than
padding capacity. Lantern's GQA uses8query/2KV heads,64features/head, window512,
RoPE10000 anddropout0. Outer layer order remains RMSNorm/residual mixer/residual
SwiGLU/RMSNorm. Tide has ranged learned retention/write gates and global RMSNorm.

Same verified Triton scan backend as the new14M hybrid, with CPU/FP64 reference.
This changes runtime versus historical eager Tide30M; timing comparisons are
therefore architecture+backend comparisons, not isolated mixer speed comparisons.
No additional routing or parallel branch. Attention KV cache and Tide recurrent
state coexist using existing CacheStorage. Total inference cache is not constant
because the inherited Lantern cache stores token history.

Reuse original larger30M tokenizer and processed datasets byte-for-byte.
Stage1:200kTinyStories/30kSQuAD training,6500updates,context512,batch8,seed42,
BF16,AdamW3e-4,warmup200,cosinefloor.1,clip1,validation1000.
Stage2:load only this hybrid's own final-stage1weights, fresh optimizer/scheduler,
QA-only6500updates,LR3e-5,warmup100. Existing full-passage eligible QA set and sampler.
Each stage:26,624,000padded input positions. Exact answer/story supervised exposure
is replayed and checked against the historical pipeline by the existing audit.
Story/QA loss and perplexity, greedy256-question EM/F1, and fixed story prompts are
evaluated before/after QA. Validation data and tokenizer identical to historical runs.
14M and30M use different tokenizer/corpus; their perplexities are not directly
comparable. Single seed, count differences, weak QA metrics limit interpretation.

Separate branches and result roots:
results/pretraining/tide-lantern-30m-v1/
results/sft/tide-lantern-30m-qa-v1/
No trained14M weights are reused. Existing train/sft/evaluation scripts remain.
External reporting wrappers reuse the prior30M runner; no new training framework.

## Completed comparison

Both hybrid stages completed6500updates/26,624,000padded positions each. Frozen source/artifacts/settings/exposure/initialization were audited.730tests and Ruff passed before training.

| Model | Params | Mixed min | Story loss / PPL before QA | QA loss / PPL before QA | EM / F1 % |
|---|---:|---:|---|---|---|
| Lantern | 29,565,952 | 20.48 | 2.055921 / 7.8140 | 6.080587 / 437.29 | 0.0000 / 1.1896 |
| Tide | 29,579,776 | 24.28 | 2.069292 / 7.9192 | 6.141600 / 464.80 | 0.0000 / 0.6689 |
| Hybrid | 28,395,520 | 18.08 | 1.990089 / 7.3162 | 6.056534 / 426.89 | 0.0000 / 0.4765 |

| Model | QA min | Story loss / PPL after QA | QA loss / PPL after QA | EM / F1 % | Story PPL ratio |
|---|---:|---|---|---|---:|
| Lantern | 21.26 | 2.564399 / 12.9928 | 5.078574 / 160.54 | 0.7812 / 2.3663 | 1.6628 |
| Tide | 24.97 | 2.764866 / 15.8769 | 5.252986 / 191.14 | 0.0000 / 1.1145 | 2.0049 |
| Hybrid | 15.63 | 2.769100 / 15.9443 | 3.766497 / 43.23 | 0.3906 / 3.0155 | 2.1793 |

Hybrid peak CUDA allocated GiB (mixed, QA): unknown for full stage; restart peak recorded separately, 2.156. Padded positions/s: 24536, 28393. Timings include loading,validation andsaving; final evaluations/samples/probes are separate.

Only one seed was tested. More attention was not tested, so this experiment does not identify an optimal mixing ratio. Model size and FFN differences prevent treating it as a pure equal-parameter ablation. QA EM/F1 remain limited evidence and are not a reasoning benchmark.

Mixed-stage timing includes a failed attempt, suspend time and checkpoint replay; it is approximate. The first attempt CUDA peak was lost. Restart-only timing/peak and the separate benchmark are preserved; do not interpret the aggregate as clean steady-state performance.

### Stability and memory

Initial: held-out-batch gradient norm 3.89103; all measured gradients and Tide outputs finite. Gate/state distributions are in the probe JSON. Bins are feature slices under global RMSNorm, not semantic heads.

After mixed training: held-out-batch gradient norm 0.96275; all measured gradients and Tide outputs finite. Gate/state distributions are in the probe JSON. Bins are feature slices under global RMSNorm, not semantic heads.

After QA: held-out-batch gradient norm 3.42690; all measured gradients and Tide outputs finite. Gate/state distributions are in the probe JSON. Bins are feature slices under global RMSNorm, not semantic heads.

Attention layers retain their KV history; this hybrid does not preserve Tide's constant-size total inference cache. Each Tide layer itself still stores one512-feature FP32 state.

### Fixed story generations

Before QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She wanted to play with it, but it was too high up.
> Lily asked her mom, "Can I climb the ball, please?" Her mom said, "Sure, but be careful." Lily

Before QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy to see Tom and his dog. They played together and had lots of fun.
> After a while, Tom's mom came home and saw the dog. She was very angry and sad. Tom and his dog were sad and scared. They went to the house and told their mom what happened. Mom was not

Before QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and scared.
> He went to his mom and said, "Mom, I lost my toy. Can you help me find it?" His mom said, "I can help you find it. Let's look together."
> They looked and looked, but they could not find the toy. They looked under the bed

After QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy teddy. One day, Lily's mom asked her to clean up her room. Lily didn't want to do it, but she knew it was important.
> As she was cleaning, she accidentally spilled on her bed. She accidentally spilled spilled on her

After QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very friendly and liked to play.
> "Hi, Tom!" said Tom. "I am sorry." Tom smiled and said, "I am sorry, Tom." They played together and had fun.
> After a while, Tom's mom came in. She saw Tom and Tom playing with his toy. Tom said,

After QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and scared. He did not want to play with anyone.
> The boy's mom saw him crying and asked what was wrong. The mom said he was sorry and he did not want to play with his toy. The boy cried and cried cried. The mom cried and cried cried.
> The mom cried and cried


## Interpretation

At14M scale the hybrid lowered story loss to1.983780 versus Tide2.010153 and
Lantern2.018320. At30M scale, mixed-pretraining story loss1.990089 is also below
Tide2.069292 and Lantern2.055921. These are within-scale, single-seed observations;
14M and30M use different tokenizers/corpora, so cross-scale loss/PPL is not comparable.

After QA, this hybrid reaches QA loss3.766497/PPL43.2284, versus Tide5.252986/PPL191.14
and Lantern5.078574/PPL160.54. Yet generated QA EM is only0.390625% (1/256), with
F1 3.015468%. Lantern had2/256exact matches, Tide0/256. A much lower answer
perplexity therefore did not yield reliably correct free generation in this test.

Story PPL rises from7.3162 to15.9443 (ratio2.1793), versus Tide2.0049 and Lantern1.6628.
The mixture improves the pretraining story metric and answer loss, but does not fix
catastrophic interference. Its final story loss2.769100 is close to Tide2.764866 and
worse than Lantern2.564399. The lower starting loss makes its relative forgetting
larger. These results do not identify the optimal attention fraction or establish
that the recurrent layers perform semantic retrieval. No additional run was started.

The CUDA failure followed system suspend/resume. Training resumed from checkpoint
6000 with optimizer/RNG/sampler restoration; the final logical exposure still matches
6500updates per stage. Mixed-stage wall time is approximate and includes interruption
and replay; the first-attempt peak is unavailable. Restart-only peak2,316,069,888bytes
and benchmark2,311,875,584bytes are separate measurements. QA-stage timing937.692s
(15.63min) and peak2,314,890,240bytes (2.156GiB) were measured directly.
