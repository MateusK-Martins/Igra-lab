# A16 plus independent compressed-gMLP blocks, no Tide

Branch `model/001-spoony-a16-separated-gmlp-14m`. Six layers, width384, unchanged FFN1024,tied vocabulary8192. Each layer: `RMSNorm → residual A16/RoPE → RMSNorm → residual CompressedGMLP → residual SwiGLU → RMSNorm`. A16 has16D Q/K/V,onehead,RoPE10000,unbounded causal history,unchangedKV cache. No Tide. One FFN per layer, not an extra FFN for the gMLP block.

Independent compressed branch: `z=W_in x`, `branch=W_out(z+gMLP(z))`; external residual adds384D x once. Bias-free384→16 and16→384 projections; same gMLP16→64/GELU/split32/LayerNorm/masked spatial matrix/product/32→16 as integrated run. Non-overlapping64-token windows, initialization and streaming/cache semantics unchanged. Additional RMSNorm has384learned gains,eps1e-6. gMLP receives the normalized full stream after attention, rather than attention's16D retrieved vector.

**10,486,752parameters** versus10,410,720 integrated. The76,032difference equals six pairs of384↔16projection weights(73,728) plus six384-gain RMSNorms(2,304). Each gMLP itself has5,840parameters in both designs. Preserve384backbone/1024FFN rather than pad to14M. Thus this comparison changes input, normalization, residual placement and projections, not only a software boundary.

Same14M pipeline: identical TinyStories tokenizer/data hashes,seed42,context256,batch8,BF16,AdamW3e-4,warmup300,9000freshupdates/18,429,440targettokens, same validation and fixed generations. Initial tensors are not explicitly paired beyond the common seed. Separate checkpoints/reports, no warm-start. Six compressed-gMLP caches2048values each; six A16KV caches32values/token. At256tokens/sample andBF16,cache120KiB excluding metadata/allocator overhead. Standalone compressed cache uses the existing serialized vector StateEntry, distinct layeridentifier. No new cache storage implementation or training stack.

Mechanism tests include explicit branch/residual reference and input/parameter gradients, float64gradcheck, causal/future gradient isolation, single-token/chunked inference and snapshots across windows, bounded gMLP memory, BF16CPU/CUDA forward/backward and cached execution, dimensions and full-model integration. Existing attention/gMLP/scan tests retained; Tide scan model test now has an explicit Tide fixture instead of depending on the configured architecture. Large checkpoints outsideGit at `/var/tmp/igra-a16-separated-gmlp-14m`. Single-seed loss does not establish reasoning/retrieval/semantic specialization. Results follow.

## Completed controlled comparison

Same data/tokenizer/settings hashes and9000updates/18,429,440actual targettokens verified. Both trained cache checks passedFP32/FP64 over105tokens across64-token windows. Sources stayed frozen during training.

| Model | Parameters | Validation loss | Perplexity | Training seconds | Peak CUDA GiB |
|---|---:|---:|---:|---:|---:|
| Integrated A16 + gMLP | 10,410,720 | 2.168635 | 8.746335 | 239.56 | 0.569 |
| Separated A16 + gMLP | 10,486,752 | 2.192682 | 8.959214 | 259.28 | 0.615 |
| A16/RoPE alone | 10,375,680 | 2.170240 | 8.760389 | 215.33 | 0.560 |
| Tide/A16 + compressed gMLP | 12,652,960 | 2.023517 | 7.564881 | 253.21 | 0.673 |

Integrated and separated each have six attention modules, six gMLPs and six FFNs; neither contains Tide. Both use RoPE10000 and the same64-token gMLP window. Standalone separated input/compression/residual/RMSNorm differences mean the result cannot isolate only block packaging. Parameter difference76,032(0.73%of integrated). Initial common seed is42, but initial tensors are not explicitly paired. No architecture tuning after completed results.

Finite initial/final held-out BF16 global gradients and six gMLP output probes passed. Spatial weight/bias/output RMS distributions are in initial-gmlp-probe.json/final-gmlp-probe.json. Benchmark throughput, full loss curves and fixed-prompt generations are in each training-report.json; final metrics use update9000, generations use the best saved checkpoint. These results do not prove reasoning or retrieval ability.

## Interpretation

Integrated gMLP is nearly tied with A16/RoPE alone: loss2.168635 versus2.170240, a difference of0.001606. This single-seed result does not demonstrate a meaningful benefit. Separated gMLP has worse loss2.192682, despite76,032additional parameters, and took259.28s versus239.56s for integrated. It used0.615GiB peak CUDA allocated memory versus0.569GiB. The separate residual/input/projection/normalization combination did not improve this run.

Both are worse than the historical Tide/A16+compressed-gMLP hybrid(loss2.023517), but that model has12,652,960parameters and a different mix of four Tide and two attention layers. It is not evidence isolating Tide independently of capacity or layer composition. No architecture was tuned after observing results. Additional seeds would be needed to assess robustness.
