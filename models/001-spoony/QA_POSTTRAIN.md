# 30M QA post-training

Separate branches preserve the original 30M models. Uses unchanged scripts.sft,
SFTDataset, training loop, optimizer, checkpoint and model runtimes. Only SFT
configuration, a dataset materializer and the existing evaluation script change.

Initialize from each model's final 6,500-update checkpoint. Retain its tokenizer,
architecture, BF16 precision and context 512. QA-only training: batch 8, seed 42,
LR 3e-5, 100 warmup updates, cosine floor 0.1, 6,500 updates. This exposes each
model to the same 52,000 questions without repetition, within 30 minutes per model.
A new optimizer/scheduler starts this stage; pretrained optimizer state is not resumed.

Pinned SQuAD v1 revision 7b6d24c440a36b6815f21b70d25016731768db1f has 87,599
training records. 86,695 survive full-passage length/dedup filtering; no answer-
centered cropping. Dataset materializer uses the already downloaded Parquet files
in data/raw/comparison-30m/squad and never retrains BPE. Dataset source/license:
https://huggingface.co/datasets/rajpurkar/squad (CC BY-SA 4.0).

The original 1,000 held-out QA examples are byte-identical; validation passages
are excluded if they overlap any official training passage. QA loss supervises
first answer plus EOS. Greedy EM/F1 uses the same 256 questions, maximum over
reference aliases, with up to 64 generated tokens. Story loss on the original
1,000 held-out stories measures forgetting. Compare final-budget checkpoints;
these are single-seed filtered-subset metrics, not leaderboard scores.

Run python -m scripts.prepare_qa_stage once, python -m scripts.sft to train,
and python -m scripts.validate_comparison to evaluate, using the model .venv.
The source, complete resolved SFT settings, initialization checkpoint hash and
data hashes are recorded by the existing manifest. Data and weights remain ignored.

## Completed results

Both models completed 6,500 SFT updates on 52,000 distinct questions (389,671
answer/EOS labels; 26,624,000 padded input positions). This is not a full epoch
over the 86,695 eligible records. Timing includes loading, periodic validation
and checkpoint saving; final generation/scoring runs separately.

| Model | Minutes | QA loss before → after | QA PPL before → after | EM % before → after | F1 % before → after | Story PPL before → after |
| --- | ---: | --- | --- | --- | --- | --- |
| Lantern | 21.26 | 6.0806 → 5.0786 | 437.29 → 160.54 | 0.00 → 0.78 | 1.19 → 2.37 | 7.81 → 12.99 |
| Flux V-only | 26.09 | 6.1891 → 5.2961 | 487.42 → 199.55 | 0.00 → 0.39 | 0.99 → 1.32 | 9.73 → 21.05 |

Answer loss uses 1,000 unchanged held-out questions; generation uses the same
256-question subset. Lantern answers 2/256 exactly, Flux 1/256, versus zero
before post-training. These small counts do not support a strong accuracy claim.
QA prediction improved modestly, but both models remain weak QA systems. Story
perplexity worsened for both, showing forgetting during QA-only tuning. The old
pretrained checkpoints remain available and the parent branches are preserved.

All tokenizer/QA data hashes and shared training/format settings match. Recorded
source hashes stayed fixed throughout both runs. The original 30,000 training
examples are a byte-identical prefix of the expanded corpus. Initialization
checkpoint hashes differ intentionally because each stage starts its own model.

Verification: 481 Lantern tests, 604 Flux tests, Ruff lint and formatting passed.
Runtime source and scripts.sft are unchanged. Results include frozen manifests,
training curves/timing, dataset provenance, exposure, QA predictions and a shared
comparison.json. Data and checkpoint weights remain local and ignored.
