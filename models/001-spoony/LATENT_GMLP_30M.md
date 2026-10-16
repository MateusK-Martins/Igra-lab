# Compressed-output gMLP / Tide–A16, established 30M pipeline

Branch `model/001-spoony-tide-a16-latent-gmlp-30m`, independently scaled from completed14M compressed-output gMLP. No window increase: gMLP still uses non-overlapping causal64-token fixed positional windows. `[Tide,Tide,A16/RoPE + latent gMLP] ×3`,9layers,width512,FFN1152,vocabulary8192/tiedhead. **26,545,264parameters** versus26,527,744 for Tide/A16; three identical5,840parameter gMLPs add17,520. Preserve FFN/backbone rather than pad to30M.

Mechanism unchanged from14M:16D Q/K/V,RoPE10000,causal SDPA retrieves16D z;`z+gMLP(z)` is projected16→512. gMLP16→64/GELU/split32/LayerNorm/spatial64×64 masked matrix/branch product/32→16. Spatial weightsNormal(0,1e-3/64),bias1. Six unchanged Tide ranged-initialized selective-write global-RMSNorm mixers use existing Triton scan. Full-width residual and FFNs stay512D. No changes to Q/K/V/cache semantics or scan. Production runtime hashes are verified identical to14M compressed-gMLP branch.

gMLP cache per layer:2,048values,4KiBBF16/8KiBFP32. Six Tide states each512FP32values. Three A16 layers cache32scalars/token. At512tokens/sample, total cache120KiB with BF16attention/gMLP history and FP32Tide, excluding metadata/allocator overhead; original Tide/A16 is108KiB. Inference cache grows with attention history; gMLP and Tide storage remain bounded.

Reuse the preceding30M pipeline without environment changes: byte-identical processed TinyStories200k/SQuAD30k mixed dataset and tokenizer;6500freshmixedupdates,context512,batch8,seed42,BF16,AdamW3e-4,warmup200. Then own final mixed weights initialize6500QA-only updates using existing eligible full-SQuAD dataset, fresh optimizer/scheduler,LR3e-5,warmup100. Eachstage26,624,000padded positions; audit exact supervised-token exposure and settings against historical30M runs. No previous experiment weights used. Evaluations:374TinyStorieswindows,1000QAanswerexamples,256greedyQA EM/F1, fixed story generations, BF16gradient/activation/gate probes, trained105-token full/cache equivalence crossing64-token boundaries.

14M and30M data/tokenizers differ, so cross-scale perplexities are not directly comparable. Primary architecture control is original30M Tide/A16. Same seed does not pair later initial tensors after different RNG consumption. Single-seed QA results do not establish reasoning/retrieval ability. QA masking makes padded-position throughput differ from supervised-answer throughput. Full pipeline, sampled stability and memory costs will be reported separately.

Checkpoints stay outsideGit in `/var/tmp/igra-tide-a16-latent-gmlp-30m/{pretraining,sft}` via ignored local symlinks. Existing training/SFT/evaluation code copied from established30M recipe; no new training stack. Reproducible manifests, audits, curves, probes and samples are saved under `results/pretraining/tide-a16-latent-gmlp-30m-v1` and `results/sft/tide-a16-latent-gmlp-30m-qa-v1`. Sources freeze before training; results follow after completion.

## Completed comparison

Both stages completed6500updates/26,624,000padded positions each. Source/artifact/settings hashes, exact supervised exposure and own-checkpoint QA initialization passed final audits.757tests/Ruff passed before training.

| Model | Parameters | Mixed min | Story loss / PPL before QA | QA loss / PPL before QA | EM / F1 % |
|---|---:|---:|---|---|---|
| Lantern | 29,565,952 | 20.48 | 2.055921 / 7.8140 | 6.080587 / 437.29 | 0.0000 / 1.1896 |
| Tide | 29,579,776 | 24.28 | 2.069292 / 7.9192 | 6.141600 / 464.80 | 0.0000 / 0.6689 |
| Tide/Lantern | 28,395,520 | 18.08 | 1.990089 / 7.3162 | 6.056534 / 426.89 | 0.0000 / 0.4765 |
| Tide/A16 | 26,527,744 | 11.23 | 2.051892 / 7.7826 | 6.139653 / 463.89 | 0.0000 / 0.8210 |
| Tide/A16 + compressed gMLP | 26,545,264 | 11.33 | 2.049449 / 7.7636 | 6.126179 / 457.68 | 0.0000 / 0.6045 |

| Model | QA min | Story loss / PPL after QA | QA loss / PPL after QA | EM / F1 % | Story PPL ratio |
|---|---:|---|---|---|---:|
| Lantern | 21.26 | 2.564399 / 12.9928 | 5.078574 / 160.54 | 0.7812 / 2.3663 | 1.6628 |
| Tide | 24.97 | 2.764866 / 15.8769 | 5.252986 / 191.14 | 0.0000 / 1.1145 | 2.0049 |
| Tide/Lantern | 15.63 | 2.769100 / 15.9443 | 3.766497 / 43.23 | 0.3906 / 3.0155 | 2.1793 |
| Tide/A16 | 11.69 | 2.646863 / 14.1097 | 5.223213 / 185.53 | 0.0000 / 0.9916 | 1.8130 |
| Tide/A16 + compressed gMLP | 11.82 | 2.738679 / 15.4665 | 5.246233 / 189.85 | 0.7812 / 1.7343 | 1.9922 |

Compressed gMLP peak allocated CUDA GiB(mixed,QA): 1.870, 1.870. Padded positions/s: 39,182, 37,549. Timings include loading/validation/saving; final evaluations/probes/generations are separate.

The direct architecture control is Tide/A16: same backbone/FFN, attention projection widths, RoPE and six Tide layers; only compressed post-retrieval gMLP is added(+17,520parameters). Later initial RNG draws change. Broader Lantern/Tide/Tide-Lantern comparisons also differ in capacity and mixer structure. Prior interrupted runs/backend differences limit timing attribution.

## Stability and fixed generations

Initial: held-out BF16 gradient norm before clipping3.981053; all gradients and measured Tide outputs finite.

gMLP outputRMS: body.9.branch.gmlp=0.134582, body.21.branch.gmlp=0.129415, body.33.branch.gmlp=0.139779. Full spatial-weight/bias and activation statistics remain adjacent.

After mixed: held-out BF16 gradient norm before clipping1.010292; all gradients and measured Tide outputs finite.

gMLP outputRMS: body.9.branch.gmlp=0.168664, body.21.branch.gmlp=0.141476, body.33.branch.gmlp=0.125942. Full spatial-weight/bias and activation statistics remain adjacent.

After QA: held-out BF16 gradient norm before clipping3.948883; all gradients and measured Tide outputs finite.

gMLP outputRMS: body.9.branch.gmlp=0.145722, body.21.branch.gmlp=0.130436, body.33.branch.gmlp=0.130736. Full spatial-weight/bias and activation statistics remain adjacent.

Trained105-token full versus7-token cached chunks passedFP32/FP64 for both final checkpoints, crossing64-token gMLP windows.

Before QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the grass. She wanted to play with it, but her mom said no. Lily was sad and didn't know what to do.
> Suddenly, she saw a little bird with a broken wing. The bird was sad

Before QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very friendly and liked to play with Tom. Tom and Tom played with the dog and had fun.
> But then, Tom saw a big dog. The dog was barking and running towards Tom. Tom was scared and ran away. He ran to the dog and tried to get his ball back. But the dog was faster

Before QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and wanted to help. He looked around and saw a big tree. He thought, "Maybe I can find a toy to play with."
> So, he went to the tree and looked for the toy. He looked under the tree, behind the tree, and in the tree. He found the toy and was

After QA, prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily and her friends were playing hide and seek. Lily was hiding hiding and her friends was laughing. Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly Suddenly

After QA, prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was barking barking and had a leash. Lily wanted to play with the dog, but Timmy didn't want to play with it. Timmy said, "No, Timmy, don't hurt me. It's not nice." Timmy felt sad and sad.
> Later that day, Timmy's mom asked him to clean up his toys

After QA, prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and sad. Lily went to him and said, "I'm sorry, Timmy. I'm sorry." Timmy smiled and hugged Lily. They played together and had fun. Timmy learned that sharing was not nice, but he was also happy.


## Interpretation

The compressed-gMLP model is nearly tied with the direct Tide/A16 control before QA: storyloss2.049449 versus2.051892. After QA, its storyloss2.738679 is worse than Tide/A16's2.646863, while QA answerloss5.246233 is also slightly worse than5.223213. Story perplexity increased by a factor of1.9922 versus1.8130 for the control. These observations do not support improved resistance to QA interference from the added gMLP in this run.

QA generation reached2exactmatches out of256(0.78125%), versus0for Tide/A16; F1rose to1.7343% versus0.9916%. These very small counts and scores do not establish a useful QA capability or a robust advantage. Teacher-forced loss, EM and F1 reflect different behavior and should not be reduced to a single winner.

The learned spatial weights moved away from their near-zero initialization and all sampled output/gradient statistics were finite, but nonzero learned temporal mixing did not produce a clear modeling improvement. No post-result tuning was performed. Multi-seed controls would be needed to distinguish small differences from initialization variability.
