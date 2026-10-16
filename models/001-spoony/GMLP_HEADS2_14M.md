# Integrated A16 / compressed gMLP: two independent temporal heads

Independent branch `model/001-spoony-a16-gmlp-heads2-14m`, based on completed single-head256-window model. Six384-wide layers,FFN1024,A16singleattentionhead16D Q/K/V,RoPE10000,no Tide. Attention retrieves16D z, then`z+gMLP(z)` before16→384projection. gMLP remains16→64/GELU/split32/32→16; same256-token fixed causal windows and learned global LayerNorm across32spatialchannels.

Only temporal filters are partitioned. Normalize all32channels first, then reshape into2heads×16channels. Each head has an independently learned256×256 lower-triangular temporal matrix and256-position bias, shared across that head's16channels. Apply batched einsum, concatenate in original channel order, multiply the matching32-channel gate branch and use existing32→16projection. No per-head normalization, router, extra output projection or cross-token mechanism. Global normalization and up/down projections still couple channel groups. Different temporal weights do not imply semantic specialization.

Two matrices/biases instead of one add65,792parameters per layer,394,752total. **11,175,264parameters** versus10,780,512singlehead. Total channels and gMLP cache stay32 and8192values/layer(16KiBBF16). At256tokens/sample, six attention+gMLP caches remain192KiB, excluding metadata/allocator overhead. Theoretical temporal MAC count stays the same because each of the two filters handles half the channels; parameter/optimizer memory increases and actual runtime must be measured. Counts include masked upper-triangle parameters.

Initialization stays independentNormal(0,1e-3/256) for each matrix,bias1. Shared seed42 does not pair later random draws when matrix count changes. Single-head default preserves original matrix shapes, cache identifiers and operations; two-head cache identifiers explicitly differ, and old single-head weights cannot silently load into new matrix shapes. Fresh training only. Cache contents remain one full32-channel normalized spatial buffer, not separate growing histories.

Same14M pipeline: identical TinyStories data/tokenizer,context256,batch8,BF16,AdamW3e-4,warmup300,9000freshupdates/18,429,440actual targettokens,same validation and generation. Sources freeze before training.771tests pass, including independent sequential head math and parameter/input gradients,float64gradcheck, identical-head equivalence to singlehead with summed temporal gradients, filter independence before down projection, shared normalization, causality, cache/snapshot equivalence,BF16CPU/CUDA513-token behavior, old checkpoint rejection and configured full-model counts/cache sizes. Existing tests retained. Checkpoints outsideGit at `/var/tmp/igra-a16-gmlp-heads2-14m`. Results follow.

## Completed comparison

All771tests/Ruff and source/artifact/settings audits passed.9000freshupdates/18,429,440targettokens. Trained273-token full versus7-token chunks crossed256boundary and passedFP32/FP64 tolerances.

| Model | Parameters | Validation loss | Perplexity | Training seconds | Peak CUDA GiB |
|---|---:|---:|---:|---:|---:|
| A16/RoPE alone | 10,375,680 | 2.170240 | 8.760389 | 215.33 | 0.560 |
| One gMLP head, window64 | 10,410,720 | 2.168635 | 8.746335 | 239.56 | 0.569 |
| One gMLP head, window256 | 10,780,512 | 2.146345 | 8.553540 | 238.40 | 0.574 |
| Two temporal gMLP heads, window256 | 11,175,264 | 2.153054 | 8.611116 | 242.36 | 0.579 |

Two heads minus one-head256 loss:+0.006709. Actual training throughput including validation/saving:76,042targettokens/s; isolated benchmark:78,521tokens/s.

Both initial/final BF16 global gradient probes and six gMLP output probes were finite. Fixed-batch per-head normalized-state/contribution RMS and temporal-gradient norms are in initial-heads-probe.json/final-heads-probe.json. Temporal-filter diagnostics by layer:

- body.1.branch.gmlp: cosine between learned masked matrices-0.689034; gated pre-down headRMS 0.415564, 0.253318; temporal gradient norms 0.007872, 0.007590.

- body.5.branch.gmlp: cosine between learned masked matrices-0.074043; gated pre-down headRMS 0.250122, 0.242243; temporal gradient norms 0.009214, 0.013514.

- body.9.branch.gmlp: cosine between learned masked matrices0.019194; gated pre-down headRMS 0.245872, 0.283320; temporal gradient norms 0.008533, 0.010912.

- body.13.branch.gmlp: cosine between learned masked matrices0.024138; gated pre-down headRMS 0.287324, 0.314944; temporal gradient norms 0.016738, 0.018499.

- body.17.branch.gmlp: cosine between learned masked matrices0.002980; gated pre-down headRMS 0.294468, 0.276027; temporal gradient norms 0.021407, 0.019346.

- body.21.branch.gmlp: cosine between learned masked matrices0.035802; gated pre-down headRMS 0.247296, 0.251238; temporal gradient norms 0.015663, 0.013349.

Distinct matrix values or channel amplitudes do not establish semantic specialization. No wider single-head control was trained in this experiment, so it cannot establish superiority over a parameter-matched widened gMLP. Full loss curve, fixed generations, manifests and probes remain adjacent. Final metrics use update9000; generations use the best saved checkpoint. Single-seed results do not establish reasoning/retrieval ability.

## Interpretation

Two heads finished at loss2.153054/PPL8.611116, compared with single-head256-window loss2.146345/PPL8.553540. The0.006709loss difference is slightly worse despite394,752extra temporal parameters. It is small and single-seed, so there is no evidence here that extra filters improve this model; it is not proof that multi-head gMLP is generally inferior. Training took242.36s versus238.40s; peak allocated CUDA0.579GiB versus0.574GiB. Cache capacity stayed unchanged.

Both heads had nonzero gated pre-down contributions and temporal gradient norms in all six layers on the held-out diagnostic batch. Learned matrix cosine similarities ranged from-0.6890to0.0358. The matrices differ numerically and there is no obvious inactive head on this sample, but sign/channel reparameterizations and shared normalization/projections prevent interpreting matrix similarity alone as functional or semantic specialization. A parameter-matched widened single-head control was not run. No architecture tuning followed these results.
