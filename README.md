# Igra-lab

A research repository for independent language models, developed through
family templates and preserved model branches.

- [Project structure and branch rules](PROJECT_STRUCTURE.md)
- [Current model](models/001-spoony/README.md)
- [Implementation roadmap](models/001-spoony/ROADMAP.md)
- [Declarative model assembly](models/001-spoony/MODEL_ASSEMBLY.md)

The current model has tested data preparation, disk-backed batches, declarative
model assembly, windowed GQA with RoPE, RMSNorm, and external KV caching. A small
four-layer decoder is configured. Next-token loss, AdamW construction, and a
warmup/cosine scheduler are implemented. Mixed precision and the training loop
are next; no real corpus or training run is configured yet.
