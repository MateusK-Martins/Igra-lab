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
