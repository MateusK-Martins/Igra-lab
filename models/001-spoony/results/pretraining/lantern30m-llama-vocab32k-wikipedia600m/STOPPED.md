# Lantern30M interim checkpoint — 118.9M tokens

Training stopped intentionally at update 7259 of 36621. This is an intermediate checkpoint, not the completed 600M-token run.

Architecture: 29,890,048 parameters; 5 Llama-style pre-norm decoder layers; width 512; SwiGLU 1280; GQA 8 query / 2 KV heads; RoPE; context 2048; tied 32,768-token embedding/output.

Validation on the separate Wikipedia validation split: loss 4.059642312956639; perplexity 57.953578127213696. Samples remain repetitive and factually unreliable.

step-00007259.pt is a full resumable checkpoint, including optimizer, scheduler, counters and RNG. tokenizer.json is the frozen 32K tokenizer. training-source.tar.gz freezes source/configuration. The exact original training source commit is 06873ba (available on the code branch). Strict resume checks require that commit, matching source, environment, paths and prepared data hashes. From the original unchanged model working directory, use `.venv/bin/python -m scripts.train --resume-latest`. Do not resume from the later results-only publication commit. Data and Python environment are not included; data preparation instructions/configs are in the source.

No training has been restarted. See SHA256SUMS for integrity checks.
