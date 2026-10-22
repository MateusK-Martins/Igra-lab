# Lantern 70M 1M-token normal instruction SFT

Stage 1 starts from the completed 600M Wikipedia checkpoint, with the original
8192-token vocabulary. Response-only supervision on javi22/simple-instruct-dataset,
pinned revision 46281e385a92468e8c037f81c8945907249f5719.
Train counts: {'examples': 9453, 'tokens': 999849, 'response_targets': 683627}. Validation: {'examples': 839, 'tokens': 99979, 'response_targets': 67034}.
One epoch, 1182 updates, seed42, BF16, context2048, batch2/accumulation4,
AdamW3e-5, 30-step warmup and cosine decay. Padding trimmed at collation.
Health watchdog and mirrored resumable checkpoints enabled.
Stage 2 must initialize from this stage's final trained weights, adding a new
<think> token and three supervised markers before each answer.
