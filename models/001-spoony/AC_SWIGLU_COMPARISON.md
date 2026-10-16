# A16-conditioned SwiGLU and matched normalization control

## Completed controlled comparison

Both fresh9000-update runs processed18,429,440targettokens with identical tokenizer/data and settings. Shared initial parameterSHA256 and exact initial BF16 fixed-batch losses matched. Production sources remained frozen; all575tests/Ruff and trained273-token full/cachedFP32/FP64 checks passed.

| Model | Parameters | Validation loss | Perplexity | Training seconds | Peak CUDA GiB |
|---|---:|---:|---:|---:|---:|
| Historical A16/RoPE, no pre-FFN RMSNorm | 10,375,680 | 2.170240 | 8.760389 | 215.33 | 0.560 |
| Matched pre-FFN RMSNorm control | 10,377,984 | 2.111566 | 8.261171 | 222.95 | 0.596 |
| A16-conditioned SwiGLU | 10,476,288 | 2.111418 | 8.259945 | 267.54 | 0.716 |

AC minus matched-control loss:-0.000148; pre-FFN-normalized control minus historical A16:-0.058674. The latter changes normalization, not conditioning.

Initial/final sampled BF16 gradients and activations were finite. AC modulation on one fixed held-out batch by layer:

- body.0: gamma mean1.000109,std0.000824,range[0.977909,1.015022],meanabsolute change0.000594,fraction near bounds0.000000,W_Cnorm2.563807.

- body.2: gamma mean1.001709,std0.013025,range[0.929609,1.076486],meanabsolute change0.010109,fraction near bounds0.000000,W_Cnorm3.657337.

- body.4: gamma mean1.000768,std0.011905,range[0.936719,1.077126],meanabsolute change0.009096,fraction near bounds0.000000,W_Cnorm3.713342.

- body.6: gamma mean1.000546,std0.009766,range[0.933335,1.079539],meanabsolute change0.007454,fraction near bounds0.000000,W_Cnorm3.744237.

- body.8: gamma mean1.004302,std0.010141,range[0.927138,1.079250],meanabsolute change0.008325,fraction near bounds0.000000,W_Cnorm4.007795.

- body.10: gamma mean1.000619,std0.008174,range[0.937904,1.074119],meanabsolute change0.005894,fraction near bounds0.000000,W_Cnorm4.208074.

Nonzero modulation indicates the learned path is active, not proof of task switching or semantic specialization. This14M phase contains onlyTinyStories training; the hypothesis about adaptation during task switching remains untested. Full curves/fixed-prompt generations, gamma/activation/gradient probes, benchmarks and manifests are retained. Final losses use update9000; samples use best saved checkpoint. Larger QA/task-switching tests would require a separate request.

## Interpretation

AC-SwiGLU loss2.111418/PPL8.259945 is almost identical to matched control2.111566/PPL8.261171: loss difference-0.000148. Most of the improvement relative to historical A16(loss2.170240) is explained by the extra pre-FFN RMSNorm, not attention-conditioned channel modulation. This is a single-seed observation, not a proof that conditioning cannot help elsewhere.

The learned path was active: W_Cnorms2.56–4.21, mean absolute gamma changes approximately0.06–1.01%by layer on the held-out batch, and no gamma values near the0.9/1.1bounds. It learned mild modulation rather than using the full allowed range. Finite sampled gradients/activations and correct cached inference show a functioning mechanism, not evidence of task-switching capability.

Conditioning added98,304parameters but increased measured training time from222.95sto267.54s(about20%) and peakCUDAallocated memory from0.596GiBto0.716GiB. Full-width conditioning activations/FP32modulation add training cost despite unchanged persistentKVstorage. No scan/attention/training optimization or post-result architecture tuning was performed.
