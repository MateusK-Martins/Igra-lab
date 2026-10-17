# Lantern 70M — English Wikipedia

Self-contained experiment branch `codex/lantern-70m-wikipedia`. Start with
[CLOUD_TRAINING.md](CLOUD_TRAINING.md) for clone, install, train, resume and
model-loading commands. No full pretraining run has been launched.

The model has **70,006,272 parameters** with vocabulary 8192, width 768, nine
layers, FFN width 2432, and a tied embedding/LM head. Attention is causal
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

## Active local 100M-token run (2026-10-17)

The active editable configuration now selects `wikipedia-en-100m`, derived from
the first 100,000,000 stored tokens of the hash-verified 600M corpus. This contains
73,218 articles (last article truncated), with the original independent 2M-token
validation set. Source revision and parent hashes are recorded in the prepared
data's `preparation-report.json`; `configs/data/wikipedia-100m-artifacts.json`
pins the new artifacts. The architecture and tokenizer are unchanged.

One shuffled pass uses 48,828 windows and 6,104 optimizer updates, covering
99,999,744 supervised target tokens. The remaining 256 stored tokens are the
shift token and incomplete trailing window. Warmup is 300 updates, validation
every 500 updates, checkpoints every 1,000 updates and at completion. Outputs
are under `results/pretraining/wikipedia-en-100m/`; `console.log` captures the
local background process. Resume with `.venv/bin/python -m scripts.train
--resume-latest` from this model directory. This is fresh pretraining, with no
previous trained 70M checkpoint present. The cloud guide describes the original
600M recipe; this section describes the active local configuration.

Training completed: 6,104 updates, validation loss 3.180789 and perplexity
24.065746. See [the run report](results/pretraining/wikipedia-en-100m/REPORT.md)
for all greedy/sampled factual probes and their limitations. They show repetition
and factual errors; this run does not establish reliable knowledge recall.
Run `.venv/bin/python -m scripts.sample` to reproduce the probes from the local
model bundle. Weights/checkpoints remain local and are excluded from Git.
