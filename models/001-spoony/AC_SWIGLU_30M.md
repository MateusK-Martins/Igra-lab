# A16-conditioned SwiGLU, 30M scale

Scale the best A16 AC-SwiGLU14M configuration, without Tide or gMLP.
14 layers, width512, SwiGLU1152, tied8192-token embedding/head; actual
29,705,216 parameters. Increased depth reaches30M while preserving the
previous30M width/FFN. Compared with9-layer30M models, depth is different.
The original AC runtime is byte-for-byte unchanged.

Each layer computes RoPE A16 on RMS-normalized input, attention residual,
pre-FFN RMSNorm and SwiGLU, modulated by gamma=1+.1*tanh(W_C z), then its
FFN residual and existing final RMSNorm. W_C:16→1152,bias-free,zero-initialized.
Q/K/V remain16D, cache32scalars/token/layer, no new persistent state.
Conditioning parameters:14*16*1152=258,048. No trained-weight warm start.

Use the established30M two-stage pipeline unchanged:200kTinyStories,
30kSQuAD, same saved tokenizer, seed42, context512,batch8,BF16,
6500mixed updates LR3e-4,warmup200; then6500QA-only updates LR3e-5,
warmup100 with a fresh optimizer initialized from this run's own mixedfinal.
Both use AdamWbetas(.9,.95),weightdecay.01,clip1,cosinefloor.1.
Each stage consumes26,624,000padded positions. Audit exact supervised tokens
and data/settings hashes against historical30M manifests before training.
QA prompts/padding are masked; padded positions differ from target tokens.

Restore the saved tokenizer and regenerate datasets only; no tokenizer retrain.
Prepared hashes must match before starting. Existing training, SFT and evaluation
scripts are reused. Evaluate TinyStories/QA loss andPPL, greedy256-question
EM/F1 and fixed story prompts both before/afterQA. Run BF16gradient/activation
probes and FP32/FP64full/cached comparisons on both final checkpoints.

One seed; increased depth and parameter differences limit architecture claims.
14M/30M use different corpora/tokenizers, so their perplexities cannot be
compared directly. The earlier14M conditioning gain over its matched
normalization control was negligible; this scale-up does not by itself isolate
conditioning from capacity. LowerQA loss is not proof of reasoning/retrieval.
