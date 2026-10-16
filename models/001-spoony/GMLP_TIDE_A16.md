# Tide–gMLP–A16, 14M pipeline

Independent branch `model/001-spoony-tide-gmlp-a16-14m`, based on the completed Tide/A16 14M branch. Six layers: `[Tide, gMLP, A16/RoPE] × 2`. Width384, outer SwiGLU1024, vocabulary8192, tied embeddings. **12,652,160 trainable parameters**, versus12,641,280 for `[Tide,Tide,A16] ×2`. No capacity padding.

The gMLP spatial gating unit follows [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050): biased384→1024, GELU, split into two512-channel branches, LayerNorm(eps1e-5) on the second, learned spatial linear mixing shared across channels, multiply the branches, biased512→384. The existing outer RMSNorm supplies input normalization; there is no additional input LayerNorm. Residual is applied once by the existing wrapper. Existing full-width outer FFNs are retained.

A learned64×64 spatial matrix is masked to its lower triangle for causality. Initialization is Normal(0,1e-3/64), spatial bias1, so the spatial gate initially approaches identity gating. Upper-triangle entries are masked and receive zero gradient; the reported parameter count includes them. The same matrix and position-specific bias are reused in non-overlapping64-token windows. Window alignment starts at sequence/cache position0 and does not shift with inference chunk boundaries. This is a causal windowed adaptation of the original fixed positional mixing, not a relative-lag/Toeplitz convolution. gMLP history resets every64tokens; Tide and attention do not reset.

Each gMLP cache uses the existing snapshot-supported StateEntry with its own GMLP-w64 layer identifier: empty key vector and flattened64×512 normalized spatial-input buffer. Only the populated current-window prefix is read, determined by total token count. Cache is inference-only and bounded, though substantially larger than Tide:32,768 scalars per gMLP layer (versus384FP32 Tide scalars). Two gMLPs, two Tide states, and two growing A16 KV caches. Old model checkpoints are not load-compatible; fresh training only.

Training remains seed42, TinyStories existing tokenizer/splits, BF16, context256, batch8, AdamW3e-4, warmup300,9000updates, same sampling/validation/generation procedures. Data and tokenizer fingerprints are compared byte-for-byte against the parent before training. Generation follows the inherited fixed-prompt evaluation with a trailing256-token context; that procedure resets window alignment when context is cropped. Streaming equivalence is independently tested with persistent absolute cache counts.

Tests:740 passing, including independent sequential mathematical and parameter/input gradient reference, float64 gradcheck, masked gradients/future isolation, chunk/single-token inference across window boundaries, snapshot restoration, bounded cache size, residual once, BF16 CPU/CUDA and1025-token zero-input stability. Existing scan and model tests retained. Completed results follow below.

## Completed run

Frozen recipe commit468abe9;9,000freshupdates/18,429,440targettokens, same verified tokenizer/data as parent. Source SHA256 hashes remained unchanged during training. All740tests and Ruff checks passed.

| Model | Parameters | Validation loss | Perplexity |
|---|---:|---:|---:|
| Tide–gMLP–A16 |12,652,160|1.993349|7.340073|
| Tide–Tide–A16 |12,641,280|2.025097|7.576844|
| Pure Tide |13,774,080|2.010153|7.464462|
| Lantern |12,440,064|2.018320|7.525675|
| Tide–Tide–Lantern |13,329,408|1.983780|7.270169|

Training wall time including validation/checkpoints:247.41s;throughput74,490targettokens/s;isolated benchmark throughput77,041tokens/s. Peak CUDA allocated memory:0.668GiB. Benchmark uses a discarded model; final run starts from scratch.

Loss difference versus Tide–Tide–A16:-0.031748. Single-seed comparison: this measures replacing two long-lived gated vector memories with local learned positional gating, with nearly equal parameter count. gMLP adds local temporal processing but removes two Tide states; any difference cannot establish that gMLP universally improves or harms hybrids. OuterFFNs and RoPEattention remain unchanged.

Both initial/final BF16 held-out gradient probes were finite. gMLP outputRMS by layer:

- body.5.branch: initial0.197648,final0.382641; final lower-triangle spatial-weightRMS0.038959.

- body.17.branch: initial0.199325,final0.362334; final lower-triangle spatial-weightRMS0.036331.

Per-Tide learned retention/write distributions, full validation curve, fixed-prompt generations, benchmarks, manifests, and trained-cache checks are in `results/pretraining/tide-gmlp-a16-14m/`. Generations use the best saved checkpoint; final validation metrics use update9000. Large checkpoints remain outsideGit in `/var/tmp/igra-tide-gmlp-a16-14m` through an ignored local symlink.

The trained model matched full-sequence and cached execution across64-token boundaries over105tokens: maxFP32logit difference1.29e-5, FP642.49e-14. The gMLP spatial weights grew substantially from near-zero initialization; finite outputRMS and gradients indicate stable learning on the sampled batch, not proof of any specific semantic role. No multiple-seed inference is possible. The same seed is retained, but replacing modules changes random-number consumption and therefore some later layer initial weights; unchanged tensors were not explicitly paired across architectures.
