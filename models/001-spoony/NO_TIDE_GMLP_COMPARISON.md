# A16 and compressed gMLP: integrated versus separate residuals

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
