# Tide 30M: matched story/QA pipeline

Branch model/001-spoony-tide-30m starts from Tide bd34980; 14M Tide is preserved.
Fresh mixed-corpus training, not resized or warm-started 14M weights.

Architecture: width512, nine layers, SwiGLU1152, tied8192 embedding/head,
global internal learned RMSNorm, sigmoid query, tanh signed candidate,
independent sigmoid retention/write gates. S=g*S_previous+i*tanh(W_V*x).
Existing residual wrapper adds sigmoid(W_Q*x)*RMSNorm(S) once.
Range u_j=j/511, j0..511; g_init=.90+.095*u_j,i_init=.20-.18*u_j.
All gate weights start zero; all feature-wise biases remain learnable.
Original parallel scan/runtime unchanged. FP32 gates/state under BF16.
One512-value FP32 recurrent state/sample/layer (2KiB), no K/matrix/attention.
Exactly29,579,776 parameters, matching original FluxV-30M. Four diagnostic
bins of128 features are not independently normalized heads.

The established30M pipeline is reused without new training infrastructure:
- Stage1: 200000 TinyStories/30000 SQuAD training records, original larger
  training-only8192 tokenizer; context512,batch8,seed42,BF16,AdamW3e-4,
  warmup200,cosinefloor0.1,clip1,6500updates. Existing shuffled ConcatDataset
  and answer-only QA labels. Same split/tokenizer/data hashes and sampler.
- Stage2: initialize only model weights from Tide's own final6500-update
  checkpoint. Fresh optimizer/scheduler; QA-only LR3e-5,warmup100,6500updates,
  batch8,context512,seed42. Same86695 eligible full-passage questions;
  exactly52000 sampled without repetition. Existing scripts.sft unchanged.
- Evaluate final-budget story loss/PPL on1000 stories, QA answer loss/PPL
  on1000 held-out questions, greedy EM/F1 on the same256 questions, max64
  new tokens. Evaluate before/after QA to measure forgetting.
- Each stage consumes26624000 padded input positions. Stage1 actual
  supervision:19748864 story labels +101560 answer/EOS labels.
  Stage2:389671 answer/EOS labels. Exposure replay verifies exact matching.
- Shared30-minute-per-stage protocol retains6500updates; wall time measured.
  Benchmark fit/reliability before full runs; no context/token reduction.

All original30M datasets/tokenizer are reused locally, not regenerated.
QA stage validation file is byte-identical to mixed-stage held-out QA.
Comparison references: Lantern29,565,952 params and FluxV29,579,776 params.
Sources/settings/artifacts/manifests are frozen and independently audited.
PPL cannot be directly compared with14M Tide's different tokenizer/corpus.
Single seed, filtered subsets, weak QA counts: not a leaderboard/reasoning
or general architecture-superiority test. EM/F1 and forgetting both reported.

Verification:703 tests/Ruff pass. Includes scaled full-context forward/
backward, recurrent/cache tests, gate range precision, exact count,
nine independent layers, original pipeline mixed-loader tests, SFT tests,
and evaluator-stage selection. Benchmarks/probes use discarded weights.

Commands from this model directory with its existing .venv:
.venv/bin/python -m scripts.benchmark
.venv/bin/python -m scripts.train
.venv/bin/python -m scripts.validate_comparison
.venv/bin/python -m scripts.sft
.venv/bin/python -m scripts.validate_comparison --sft

New result roots are results/pretraining/tide-30m-v1 and
results/sft/tide-30m-qa-v1. Large datasets/checkpoints remain local/ignored.
Historical experiment documentation below README is retained for provenance.

## Completed results

Each stage completed 6,500 updates / 26,624,000 padded positions. Supervised exposure was independently replayed and matched against the historical pipeline. Source hashes stayed fixed throughout both stages; tokenizer/data/settings and initialization checkpoint hashes were verified.

| Model | Params | Mixed minutes | Story loss / PPL before QA | QA loss / PPL before QA | EM / F1 % before QA |
|---|---:|---:|---|---|---|
| Lantern | 29,565,952 | 20.48 | 2.055921 / 7.8140 | 6.080587 / 437.29 | 0.0000 / 1.1896 |
| FluxV | 29,579,776 | 25.38 | 2.275054 / 9.7284 | 6.189136 / 487.42 | 0.0000 / 0.9868 |
| Tide | 29,579,776 | 24.28 | 2.069292 / 7.9192 | 6.141600 / 464.80 | 0.0000 / 0.6689 |

| Model | QA-only minutes | Story loss / PPL after QA | QA loss / PPL after QA | EM / F1 % after QA | Story PPL ratio |
|---|---:|---|---|---|---:|
| Lantern | 21.26 | 2.564399 / 12.9928 | 5.078574 / 160.54 | 0.7812 / 2.3663 | 1.6628 |
| FluxV | 26.09 | 3.046811 / 21.0481 | 5.296061 / 199.55 | 0.3906 / 1.3171 | 2.1636 |
| Tide | 24.97 | 2.764866 / 15.8769 | 5.252986 / 191.14 | 0.0000 / 1.1145 | 2.0049 |

Tide peak VRAM (mixed,QA-only): unavailable; benchmark3.259GiB separately measured, 3.259GiB. Throughput: 18276, 17773 padded positions/s. Timing includes loading,periodic validation andsaving; final evaluations/probes/generations are separate. Historical models were timed in their own runs.

Mixed-stage time/throughput are approximate: recovered from log creation and final checkpoint mtime after a reporting-only JSON error. Actual mixed-stage CUDA peak was lost; the separate benchmark used3.259GiB. QA-stage timing/peak were persisted directly. Both stages completed exactly once; checkpoints/weights and evaluations are unaffected.

### Gate/state stability

Initial/final probes use the same first held-out story batch under BF16. The post-QA probe therefore diagnoses story activations/gradients after tuning, not an aggregate QA gradient measurement. Four bins are contiguous128-feature slices under global normalization, not independent head norms.

#### Initial

Gradient norm 4.16580; all sampled gradients/states/outputs finite.

| Bin | Retention mean | Retention token std | Write mean | Write token std | Output RMS |
|---|---:|---:|---:|---:|---:|
| 1 | 0.91181 | 0.00000 | 0.17763 | 0.00000 | 0.52300 |
| 2 | 0.93560 | 0.00000 | 0.13254 | 0.00000 | 0.51085 |
| 3 | 0.95940 | 0.00000 | 0.08746 | 0.00000 | 0.51049 |
| 4 | 0.98319 | 0.00000 | 0.04237 | 0.00000 | 0.52791 |

#### After mixed training

Gradient norm 0.96235; all sampled gradients/states/outputs finite.

| Bin | Retention mean | Retention token std | Write mean | Write token std | Output RMS |
|---|---:|---:|---:|---:|---:|
| 1 | 0.75605 | 0.17529 | 0.20949 | 0.12606 | 0.54081 |
| 2 | 0.81481 | 0.14172 | 0.14009 | 0.09125 | 0.45774 |
| 3 | 0.87535 | 0.11103 | 0.08245 | 0.05926 | 0.35523 |
| 4 | 0.93722 | 0.08582 | 0.03602 | 0.03118 | 0.34449 |

#### After QA-only training

Gradient norm 4.42893; all sampled gradients/states/outputs finite.

| Bin | Retention mean | Retention token std | Write mean | Write token std | Output RMS |
|---|---:|---:|---:|---:|---:|
| 1 | 0.76505 | 0.17119 | 0.21236 | 0.12264 | 0.54682 |
| 2 | 0.82298 | 0.13848 | 0.14122 | 0.08881 | 0.46310 |
| 3 | 0.88164 | 0.10889 | 0.08420 | 0.05805 | 0.37192 |
| 4 | 0.94062 | 0.08438 | 0.03660 | 0.03027 | 0.37166 |

### Story samples and QA predictions

Fixed story prompts use greedy generation,64 new tokens, recomputed last512-token context, matching the inherited story sample procedure. QA predictions use the inherited constant-state cached decoder and filtered prompt-length cap. Full256 QA predictions/aliases/scores are saved for each stage. Neither generation nor perplexity establishes reasoning/retrieval.

#### Before QA

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She wanted to play with it, but her mom said it was too dangerous.
> Lily was sad, but she listened to her mom. She went to the park and saw a boy playing with a ball

#### Before QA

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was brown and had a long tail. Tom wanted to play with the dog, but the dog was scared.
> Tom saw a dog and ran after the dog. The dog barked and barked, but Tom was too scared. He ran away and hid behind a tree.
> Tom saw a dog and ran after the dog.

#### Before QA

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and scared. He did not know what to do.
> He went to his mom and said, "Mom, I lost my toy. I can't find it." His mom said, "Don't worry, I will help you find it." They looked and looked, but they could not find the toy.

#### After QA

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her mommy. One day, Lily's mommy asked her to clean up her toys. Lily didn't want to do it, but mommy reminded her that she didn't want to.
> After dinner, Lily's mommy asked her if she wanted to play with her toys. Lily didn

#### After QA

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was brown and had a long tail. Tom was very happy.
> "Can I play with your dog?" Tom asked.
> "No, you can't. This is my dog. He is my friend. He is my friend." Tom said.
> "No, he is mine. I want to play with him

#### After QA

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and wanted to cry.
> The boy saw Timmy and said, "I'm sorry, I'm sorry. Can't you fix your toy?" The boy said, "I'm sorry, but I can't fix your toy." The boy said, "I will help you find your toy."
> The boy was

## Interpretation

Tide's mixed-stage story loss2.069292 is close to Lantern2.055921 and
substantially lower than FluxV2.275054. Its story perplexity is18.6% below
FluxV and1.35% above Lantern in this run. Tide and FluxV have exactly matched
parameter counts and FFN width; Lantern has slightly fewer parameters and
allocates more capacity to its FFN. This is not an initialization-matched
comparison across architectures; RNG consumption differs.14M perplexities
use a different tokenizer/corpus and cannot establish a scaling trend here.

QA answer loss improves after post-training for all models. Tide sits between
Lantern and FluxV on teacher-forced QA loss, both before and after tuning.
Generated QA remains weak: Tide answers0/256 exactly, FluxV1/256 and
Lantern2/256 after tuning. Tide's F1 rises0.67%→1.11%, below FluxV1.32%
and Lantern2.37%. The tiny exact-answer counts do not justify strong ranking
claims. Lower answer perplexity than FluxV did not translate into better
EM/F1, demonstrating why both measurements are necessary.

Tide's story PPL rises7.9192→15.8769 (2.0049x), and story loss increases
0.695575 during QA-only tuning. Forgetting is substantial. Its relative
deterioration is smaller than FluxV2.1636x, but larger than Lantern1.6628x.
Absolute post-QA story quality also lies between them. This protocol does
not eliminate catastrophic interference; it measures one task shift under
one fixed recipe. It does not isolate a particular gate or normalization
as the cause of forgetting.

Tide has one512-element state per layer versus FluxV's two512-element states;
both preserve constant-size inference memory. The existing eager scan still
materializes training intermediates, so constant inference state does not
mean constant training memory. Actual QA-stage peak allocation was3.259GiB.
Mixed-stage benchmark used3.259GiB; actual training peak was not captured
after the reporting wrapper failed. Mixed duration24.28min is approximate;
QA duration24.97min is directly timed. Both fit the original30min per stage,
and both completed the full6500updates with no retraining or token reduction.

Initial/post-mixed/post-QA held-out story gradient probes were finite, and
all saved validation losses were finite. Learned per-feature gates remain
input-dependent; activation/gate distributions are diagnostics, not evidence
of semantic memories. Fixed story samples and all256 QA predictions are
saved for inspection. Data/tokenizer/settings/exposure/initial-weight hash
checks passed across stages and against the historical comparison.

Frozen recipe:4987db6. Previous14M Tide and the older30M branches remain
unchanged. Large datasets and checkpoints remain local and ignored.
