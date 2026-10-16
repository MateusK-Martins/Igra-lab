# A16-conditioned SwiGLU,14M pipeline

Branch `model/001-spoony-a16-ac-swiglu-14m`. Plain A16/RoPE, no Tide or gMLP. Six layers,width384,FFN1024,tied8192vocabulary. A16Q/K/V remain16D,RoPE10000,causal softmax/SDPA,KV32scalars/token/layer. Attention returns transient16D z;`r=x+W_Oz`. Then`u=RMSNorm(r)`, `h=SiLU(W_Gu)*W_Uu`, `gamma=1+0.1*tanh(W_Cz)`, `y=r+W_D(h*gamma)`. Retain existing final RMSNorm per layer. No additional residual wrapper around the whole layer, no detached z, no mutable activation handoff, no new persistent state/cache.

Use1024FFNhidden from14M recipe, not1152from30M example. Bias-free conditioning16→1024 adds16,384weights/layer,98,304total. W_C initialized exactlyzero, so gamma=1 initially. Compute tanh/affine gamma inFP32 forFP16/BF16 inputs(preserveFP64 references); multiply hidden activations by that gamma before existing autocast down projection. Mathematical bounds[0.9,1.1]. Zero-init tests verify exact matched-control outputs inFP32/FP64/BF16 and nonzero first-step conditioning gradients.

Historical A16 recipe feeds r directly to its FFN. The user's formula explicitly adds pre-FFN RMSNorm, adding384gains/layer,2,304total. Therefore run a matched unconditioned pre-FFN-normalized control: **10,377,984parameters**, versus historical A16 **10,375,680**. AC-SwiGLU has **10,476,288parameters**. Treat control comparison as the effect of conditioning; comparison with historical A16 includes normalization. Shared embeddings,Q/K/V/O,FFNs and norms initialize identically across AC/control. Attention construction retains original forked RNG behavior; creation of zero-initialized W_C is also RNG-forked, so later shared tensors do not shift.

Implementation minimally exposes existing attention's `retrieve()` result; its default forward remains `out_proj(retrieve())` for all existing attention modes. One declarative ACSwiGLU block holds the local z across attention and FFN and owns both residual additions. Reuses existing RMSNorm,SwiGLU,attention,cache,training and evaluation. Old vector/matrix attention tests use explicit original architecture fixtures; they are not deleted. New tests cover scalar attention/formula/parameter+input gradient reference,float64gradcheck,zero-init learning signal,bounds,snapshot/chunk/single-token equivalence,causality,BF16CPU/CUDA,parameter/cache counts,sharedinitialization and no retained retrieval buffers.575tests and Ruff pass before training.

Protocol: same30kTinyStories/256validationdocuments, preserved8192tokenizer,seed42,context256,batch8,BF16,AdamW3e-4,warmup300,9000freshupdates/18,429,440actual targettokens,same validation/fixed prompts. Rebuilt deleted data with existing scripts; all seven fingerprints exactly match historical A16 manifest. Data/order seeds remain independent of model initialization. No warm-start. Data/intermediates will be removed after runs; preserve each final checkpoint/tokenizer in central artifact folder, per user's storage preference. Single seed; small differences do not establish reasoning/retrieval or task-switching ability. This phase measures TinyStories language modeling, not controlled task switching. Results follow.

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
