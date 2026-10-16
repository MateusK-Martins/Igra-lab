# Tide/Lantern hybrid 30M pipeline

[Tide,Tide,Lantern] repeated three times: six Tide and three Lantern mixers,
width512, nine layers, SwiGLU1152, tied8192 embedding/head initstd.02.
28,395,520parameters, 4.00% below Tide29,579,776. Preserve Tide's FFN rather than
padding capacity. Lantern's GQA uses8query/2KV heads,64features/head, window512,
RoPE10000 anddropout0. Outer layer order remains RMSNorm/residual mixer/residual
SwiGLU/RMSNorm. Tide has ranged learned retention/write gates and global RMSNorm.

Same verified Triton scan backend as the new14M hybrid, with CPU/FP64 reference.
This changes runtime versus historical eager Tide30M; timing comparisons are
therefore architecture+backend comparisons, not isolated mixer speed comparisons.
No additional routing or parallel branch. Attention KV cache and Tide recurrent
state coexist using existing CacheStorage. Total inference cache is not constant
because the inherited Lantern cache stores token history.

Reuse original larger30M tokenizer and processed datasets byte-for-byte.
Stage1:200kTinyStories/30kSQuAD training,6500updates,context512,batch8,seed42,
BF16,AdamW3e-4,warmup200,cosinefloor.1,clip1,validation1000.
Stage2:load only this hybrid's own final-stage1weights, fresh optimizer/scheduler,
QA-only6500updates,LR3e-5,warmup100. Existing full-passage eligible QA set and sampler.
Each stage:26,624,000padded input positions. Exact answer/story supervised exposure
is replayed and checked against the historical pipeline by the existing audit.
Story/QA loss and perplexity, greedy256-question EM/F1, and fixed story prompts are
evaluated before/after QA. Validation data and tokenizer identical to historical runs.
14M and30M use different tokenizer/corpus; their perplexities are not directly
comparable. Single seed, count differences, weak QA metrics limit interpretation.

Separate branches and result roots:
results/pretraining/tide-lantern-30m-v1/
results/sft/tide-lantern-30m-qa-v1/
No trained14M weights are reused. Existing train/sft/evaluation scripts remain.
External reporting wrappers reuse the prior30M runner; no new training framework.
