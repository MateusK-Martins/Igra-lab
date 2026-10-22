# Normal-instruction Lantern70M -> 1M TinyStories tokens

Initialize from the normal 1M-instruction SFT checkpoint, not the think model.
Retain original 8192-token tokenizer and 70,006,272 parameters. Plain TinyStories
text + EOT, packed 2048-token causal LM windows with supervision at every position.
1,000,000 stored train tokens; 999,424 usable targets, 61 updates. Independent
50,000-token validation slice. Source order; hash deduplication across splits.
AdamW3e-5, BF16, batch2/accumulation4, seed42, 5-step warmup/cosine.
Existing device-health save-and-stop monitor is enabled. Atomic periodic/final
checkpoints are mirrored outside the worktree; retain latest one in each place.
See data/processed/tinystories-1m/preparation-report.json for dataset provenance.
Run `.venv/bin/python -m scripts.train`, or `--resume-latest` for interruptions.
