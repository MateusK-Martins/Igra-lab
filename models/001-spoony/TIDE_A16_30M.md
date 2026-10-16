# Tide / A16 — established 30M pipeline

Branch model/001-spoony-tide-a16-30m. Scaled from the 14M Tide/A16 design using
the existing Tide/Lantern 30M backbone and pipeline. [Tide,Tide,A16] repeated
three times: nine width-512 layers, six unchanged Tide selective-write/global
RMSNorm mixers and three single-head A16 Q/K/V attention mixers. FFN width1152,
tied vocabulary8192 embedding/head initstd.02. A16 has 16D Q/K/V, bias-free
projections, RoPE10000, full causal history and dropout0. Output16-to-512.
Outer layer order stays RMSNorm -> residual mixer -> residual SwiGLU -> RMSNorm.
Existing GQA runtime with one query/KV head implements A16, and the existing
Tide Triton scan is unchanged. No new runtime/training stack or environment.

26,527,744 actual parameters, 1,867,776 below the prior 28,395,520-parameter
Tide/Lantern hybrid. Preserve FFN/backbone rather than pad to an exact30M.
Each Tide layer has512FP32 recurrent values/sample; A16 caches32scalars/token/
layer. Total hybrid inference memory grows with attention history. Snapshot and
cached offset handling reuse the existing tested mixed-cache implementation.

Copy previous30M tokenizer and processed datasets byte-for-byte; no regeneration.
Stage1:200kTinyStories/30kSQuAD,6500updates,context512,batch8,seed42,BF16,
AdamW3e-4,betas(.9,.95),eps1e-8,wd.01,warmup200,cosinefloor.1,clip1,val1000.
Stage2:initialize only from this model's own final mixed checkpoint, fresh
optimizer/scheduler, QA-only6500updates,LR3e-5,warmup100. Same full-passage
eligible QA data, sampler and evaluation as earlier30M experiments.
Each stage processes26,624,000padded positions. Audit exact supervised exposure
against the prior pipeline; it differs from padded positions because QA masks
prompt/padding targets. Same story/QA held-out losses and greedy256-question
EM/F1 before/after QA, plus fixed story generations and BF16 stability probes.
14M/30M use different tokenizer/corpus, so cross-scale perplexities are not directly
comparable. Single seed and parameter/FFN differences limit architecture claims.

733 tests and Ruff lint/format checks pass before training. Coverage includes
scaled architecture/counts/gate initialization, full/chunk cached logits,
mixed-cache snapshots and sizes, future-gradient isolation, BF16, Triton scan
forward/backward parity, plus established data/training/SFT/evaluation tests.
Training code, tokenizer, datasets, schedule and evaluation are reused.
External wrappers only capture measurements, exposure and comparisons.

Results: results/pretraining/tide-a16-30m-v1 and
results/sft/tide-a16-30m-qa-v1. Ignored checkpoint links store large files under
/var/tmp/igra-tide-a16-30m/{mixed,qa} on the root partition because home lacks
space. No previous checkpoint/data is deleted. /var/tmp may be cleaned by system
policy, so retain/move these artifacts before temporary-file cleanup. Reports,
configs, manifests, probes and QA predictions remain versioned inside this branch.
The branch is pushed after both stages and reporting complete.

## Completed comparison

Both stages completed6500updates/26,624,000padded positions each. Both-stage source/artifact/settings hashes, exact supervised exposure and own-checkpoint QA initialization passed the final audit.733tests/Ruff passed before training.

| Model | Parameters | Mixed min | Story loss / PPL before QA | QA loss / PPL before QA | EM / F1 % |
|---|---:|---:|---|---|---|
| Lantern | 29,565,952 | 20.48 | 2.055921 / 7.8140 | 6.080587 / 437.29 | 0.0000 / 1.1896 |
| Tide | 29,579,776 | 24.28 | 2.069292 / 7.9192 | 6.141600 / 464.80 | 0.0000 / 0.6689 |
| Tide/Lantern | 28,395,520 | 18.08 | 1.990089 / 7.3162 | 6.056534 / 426.89 | 0.0000 / 0.4765 |
| Tide/A16 | 26,527,744 | 11.23 | 2.051892 / 7.7826 | 6.139653 / 463.89 | 0.0000 / 0.8210 |

| Model | QA min | Story loss / PPL after QA | QA loss / PPL after QA | EM / F1 % | Story PPL ratio |
|---|---:|---|---|---|---:|
| Lantern | 21.26 | 2.564399 / 12.9928 | 5.078574 / 160.54 | 0.7812 / 2.3663 | 1.6628 |
| Tide | 24.97 | 2.764866 / 15.8769 | 5.252986 / 191.14 | 0.0000 / 1.1145 | 2.0049 |
| Tide/Lantern | 15.63 | 2.769100 / 15.9443 | 3.766497 / 43.23 | 0.3906 / 3.0155 | 2.1793 |
| Tide/A16 | 11.69 | 2.646863 / 14.1097 | 5.223213 / 185.53 | 0.0000 / 0.9916 | 1.8130 |

Tide/A16 peak allocated CUDA GiB (mixed, QA): 1.861, 1.861. Padded positions/s: 39,514, 37,957. Training timings include loading, validation and saving; final evaluations/probes/generations are separate. Historical Tide/Lantern mixed timing includes interruption/replay; historical Tide uses a different scan backend.

Same width512/FFN1152 and six-Tide/three-attention pattern as the earlier hybrid, with A16 replacing GQA. This reduces attention weights and cache, while also changing attention-head count/width, total parameters and subsequent initial RNG draws. It is not an exact parameter/initial-weight-matched ablation. QA loss and EM/F1 measure different behaviors; low answer loss alone does not prove factual retrieval or reasoning. Single-seed outcomes do not establish general superiority.

Each A16 layer caches32scalars/token instead of256 for prior GQA; six Tide layers each retain512FP32values. At512tokens/sample with BF16KV and FP32recurrent state, total cache is108KiB versus780KiB for the prior hybrid, excluding metadata/allocator overhead. Total inference cache still grows with token history.

## Stability

Initial: held-out BF16 gradient norm before clipping 3.797308; all gradients and measured Tide outputs finite. Full gate/state/readout distributions remain in probes; feature bins under global RMSNorm do not prove semantic specialization.

After mixed: held-out BF16 gradient norm before clipping 1.020569; all gradients and measured Tide outputs finite. Full gate/state/readout distributions remain in probes; feature bins under global RMSNorm do not prove semantic specialization.

After QA: held-out BF16 gradient norm before clipping 3.103808; all gradients and measured Tide outputs finite. Full gate/state/readout distributions remain in probes; feature bins under global RMSNorm do not prove semantic specialization.

Trained full-sequence versus 7-token cached chunks also passed in FP32 and FP64 for both final checkpoints. Per-stage trained-cache-equivalence.json records numerical errors.

## Fixed story generations

Before QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She wanted to play with it, but it was too high up.
> Lily's friend, Timmy, saw her and said, "Don't worry, Lily. I will help you." They went to

Before QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was big and brown. It had a long tail and a long tail. Tom and Tom were scared of the dog. They ran to the dog and tried to pet it. But the dog was faster and bit Tom's leg. Tom cried out in pain.
> "Help, help!" Tom cried. "The dog is

Before QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and scared. He wanted to find his toy, but he could not find it.
> He looked under the bed, in the closet, and in the closet, but he could not find it. He was very sad. He looked under the bed, in the closet, and even in the closet. He could

After QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big tree with lots of leaves. She wanted to climb the tree, but she was scared. Suddenly, she saw a squirrel and she ran to it. She climbed up and slid slid. Timmy was happy to see her and they played together.

After QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was brown and had a long tail.
> "Can we play with the dog?" Lily asked.
> "Sure, but be careful. The dog is not friendly. He might bite us," Lily said.
> "Can we pet him?" Lily asked.
> "No, we can't pet him. He might bite

After QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and wanted to help him. He looked everywhere for the toy, but could not find it. Suddenly was very sad.
> The boy saw the toy and smiled. He said, "I will help you find your toy." The boy was happy and thanked his toy. They played together and had lots. The boy


## Interpretation

Before QA, Tide/A16 story loss 2.051892 is close to
historical Lantern/Tide, but worse than the larger-attention hybrid's
1.990089. A16 reduces cached attention storage by
8x per token/layer and removes1,867,776parameters; this run does not show a
quality advantage from that compression.

After QA, answer loss 5.223213 is worse than the
prior hybrid's 3.766497. Greedy EM remains0%, and
F1 is only 0.9916%. These weak generation results
limit claims about useful question answering even though answer perplexity fell.
Story perplexity rises by 1.8130x, versus
2.1793x for the prior hybrid. That smaller increase
cannot establish better resistance to interference: this model also fitted the
QA objective less well. Both task improvement and story degradation must be
considered together. Single seed, parameter differences and changed attention
initialization prevent attributing these outcomes solely to latent width.
