# T16 70M — English Wikipedia

Self-contained experiment branch `codex/t16-70m-wikipedia`. Start with
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) for clone, install, train, resume and
model-loading commands. No full pretraining run has been launched.

The model has **69,725,184 parameters** with vocabulary 8192, width 768, nine
layers, FFN width 2368, and a tied embedding/LM head. Attention is causal
with no sliding-window bound. Training uses 2048-token windows, microbatch 2,
four accumulation batches, BF16 and one full pass over the 600M-token corpus.
Embedding initialization is scaled from 0.02 at width 512 to
`0.02 * sqrt(512/768)` at width 768.

The selected data is regular English Wikipedia, with 600M stored training tokens
and a separate 2M-token validation slice. QA mixing is disabled. See
[WIKIPEDIA_600M.md](WIKIPEDIA_600M.md) for provenance, deduplication and token counts.

Editable settings live under `configs/` by concern. Implementation lives under
`src/`. See [MODEL_ASSEMBLY.md](MODEL_ASSEMBLY.md) and
[CHECKPOINTS.md](CHECKPOINTS.md) for construction and persistence contracts.
