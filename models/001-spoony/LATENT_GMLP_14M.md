# Tide–A16 with gMLP on retrieved compressed vectors

Independent branch `model/001-spoony-tide-a16-latent-gmlp-14m`; previous experiments preserved. Six full-width384 layers with FFN1024, `[Tide,Tide,A16/RoPE + latent gMLP] ×2`. The two standalone full-width gMLP mixers are moved inside attention and replaced by Tide in their previous positions. Thus this has four Tide states, versus two in the previous full-width gMLP experiment; that is a comparison limitation. The more direct mechanism control is the original four-Tide/two-A16 model.

Each A16 first computes unchanged16D Q/K/V projections, RoPE10000 and causal SDPA retrieval `z[B,T,16]`. Then `z_processed=z+gMLP(z)`, followed by the existing bias-free16→384 output projection. The existing outer wrapper adds the full-width input once. Neither queries, keys, values, nor their cache representations pass through gMLP. All surrounding states/FFNs stay full width.

The reused fixed-window gMLP has biased16→64 + GELU, split into32-channel branches, LayerNorm(eps1e-5) on the spatial branch, learned64×64 lower-triangular spatial mixing shared across channels, spatial bias1, branch multiplication, biased32→16. Matrix initialization Normal(0,1e-3/64), as before. Windows are non-overlapping64-token windows aligned to absolute sequence/cache positions. Attention and Tide retain history across gMLP window boundaries. This is a causal windowed adaptation of original fixed positional gMLP, not a relative-lag convolution.

Each gMLP adds **5,840parameters** (versus596,416 for full-width gMLP). Total model **12,652,960parameters**, original Tide/A16 **12,641,280** (+11,680); previous standalone-gMLP hybrid **12,652,160** (+800). Counts include masked upper-triangle parameters. No artificial parameter padding or FFN changes.

Per-gMLP cache:64×32=2,048values,4KiB BF16/8KiB FP32,16× smaller than prior full-width gMLP cache. Standard A16KV remains32values/token/layer. Four TideFP32states384values each. Each enabled attention definition supplies its KV entry plus a separately identified GMLP StateEntry; assembler flattens this small list. Snapshots use existing format1 KV/vector serialization and verify aligned counts before forward. Default attention disables gMLP and preserves historical behavior. Fresh checkpoints only; no old architecture checkpoint reinterpretation.

Controlled protocol: identical TinyStories artifacts/tokenizer hashes, seed42, context256,batch8,BF16,AdamW3e-4,warmup300,9000updates,18,429,440actual targettokens, inherited validation and fixed-prompt generation. Same seed but later initialization tensors are not explicitly paired when architecture construction changes RNG consumption. Generation follows inherited trailing256-token context evaluation, resetting gMLP alignment when the context is cropped; persistent-cache behavior is checked independently.

Architectural tests cover explicit sequential attention→spatial gate→projection math and parameter/input gradients, float64 joint gradcheck, causality, recovered baseline with zero gMLP output, single-token/chunked execution and snapshots across windows, cache sizes/counts, early rejection of mismatched positions, BF16 CPU/CUDA training/inference and old attention tests. No expensive training until the entire suite and Ruff pass. Data/checkpoints remain outside Git; checkpoint storage `/var/tmp/igra-tide-a16-latent-gmlp-14m` via ignored local symlink. Results follow after completion.

## Completed comparison

Frozen recipe ca430f2; fresh9,000updates/18,429,440targettokens. All source hashes and shared artifact fingerprints verified. All754tests, Ruff lint and formatting passed.

| Model | Parameters | Validation loss | Perplexity |
|---|---:|---:|---:|
| Tide/A16 + compressed gMLP |12,652,960|2.023517|7.564881|
| Tide/full-width gMLP/A16 |12,652,160|1.993349|7.340073|
| Original Tide/Tide/A16 |12,641,280|2.025097|7.576844|
| Lantern |12,440,064|2.018320|7.525675|
| Pure Tide |13,774,080|2.010153|7.464462|

Training time253.21s including validation/saving; throughput72,784actual targettokens/s; isolated benchmark75,940tokens/s. Peak CUDA allocated0.673GiB. Benchmark updates are discarded; main run initializes from scratch.

Loss difference versus original Tide/A16:-0.001580; versus prior full-width gMLP hybrid:+0.030168. The first is the more controlled architecture comparison; the second also changes the number of Tide layers. Same-seed settings do not ensure paired initial tensors after different module construction. Treat all results as single-seed observations rather than evidence of general architectural superiority.

Initial/final BF16 gradient probes were finite. The trained model's105-token full/cached execution crossed64-token windows and passedFP32/FP64 tolerances. gMLP activation diagnostics on one fixed validation batch:

- body.9.branch.gmlp: outputRMS 0.123182 → 0.168700; spatial-weight lower-triangleRMS 0.000011 → 0.010553.

- body.21.branch.gmlp: outputRMS 0.121183 → 0.121490; spatial-weight lower-triangleRMS 0.000011 → 0.009479.

Full validation curve and fixed-prompt generations are in training-report.json; final loss uses update9000, generations use the best saved checkpoint. Learned Tide retention/write statistics and global parameter gradients are in initial-probe.json/final-probe.json. Cache checks, manifests and benchmark are adjacent. Finite sampled statistics do not establish long-range retrieval or semantic specialization.

## Interpretation

Compressed gMLP was nearly tied with original Tide/A16 (loss improvement0.00158) and worse than the standalone full-width gMLP hybrid by0.03017. A difference this small against the direct control is not convincing evidence of a benefit in a single-seed comparison. Spatial weights moved away from near-zero initialization, but learning a nonzero filter did not recreate the full-width result. A plausible hypothesis is that16D retrieved representations limit what temporal gating can transform; however, moving gMLP also restores two Tide layers and changes its ordering relative to attention, so these results do not isolate compression width alone.

The gMLP-specific parameter/cache savings are substantial, but whole-model measured runtime and VRAM did not improve meaningfully: unchanged full-width FFNs, embeddings and Tide/attention dominate this small model. No architecture tuning was performed after the completed run.
