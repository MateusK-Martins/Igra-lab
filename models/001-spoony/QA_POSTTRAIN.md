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
