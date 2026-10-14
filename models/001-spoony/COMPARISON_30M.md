# 30M story / QA comparison

Uses the existing model, `scripts.train`, training loop, optimizer, token stores,
SFTDataset, DataLoader builder and validator. No new runtime or config classes.
The only training-script change concatenates the existing story and QA datasets
and records QA file hashes/format in the existing manifest.

Both models: width 512, 9 layers, vocabulary 8192, context 512, batch 8,
6,500 updates, BF16, seed 42, LR 3e-4, 200 warmup updates. Lantern uses GQA
and FF width 1408 (29,565,952 parameters); Flux V-only uses token-dependent
value decay, ordinary cumulative keys, original RMSNorm and FF width 1152
(29,579,776 parameters). The scan remains parallel.

Prepared data is identical in both checkouts: 200,000 TinyStories training
stories / 1,000 validation stories and 30,000 SQuAD training questions / 1,000
validation questions. Shared tokenizer is trained on training data only.
SQuAD examples retain full passages; overlength examples are excluded and
selection uses seed 42. Source hashes and selection counts are recorded in
`data/processed/comparison-30m/preparation-report.json` (ignored dataset artifact).
Dataset sources: https://huggingface.co/datasets/roneneldan/TinyStories and
https://huggingface.co/datasets/rajpurkar/squad .

ConcatDataset is shuffled by the existing seeded loader. QA examples supervise
only answers plus EOS using the existing SFTDataset. Sampling follows dataset
sizes, without a custom batch ratio. Both models consume identical shuffled
examples. TrainingState.tokens counts padded input positions, not supervised
labels. 6,500 updates process 26,624,000 positions. The 30-minute budget applies
per model; the common update count was selected using the slower model timing.

Run `.venv/bin/python -m scripts.train`, then
`.venv/bin/python -m scripts.validate_comparison`. The latter reports final
checkpoint story loss/perplexity, answer-only QA loss/perplexity, and greedy
QA EM/F1 on 256 held-out examples, capped at 64 generated tokens. This filtered,
single-seed comparison is not a full SQuAD leaderboard evaluation. PPL cannot
be compared directly with older runs using a different tokenizer.

The earlier expanded implementation was reverted and retained only in local
archive branches. Its interrupted training run is excluded from this comparison.
