# Lantern30M Llama-style model

Completed: 29,890,048-parameter configuration, true pre-norm residual branches,
final RMSNorm, tied 32K vocabulary, frozen training-only 32K tokenizer,
2,048-token context retained, CPU/cache/backward tests and CUDA smoke.

No pretraining launched. Next training work requires retokenizing the corpus
with the new tokenizer; 8K token IDs and old 70M weights are incompatible.

Full 600M-token training authorized. Active recipe is documented in README.md.
Preparation retokenizes pinned Wikipedia with 32K BPE. Health monitoring,
checkpoint mirroring, and 494 tests are verified before launch.
