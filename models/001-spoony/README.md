# Lantern 70M — full 600M-token Wikipedia run

Fresh original architecture: 70,006,272 parameters, width 768, nine layers,
SwiGLU width 2432, tied 8,192-token embedding/head. No QAT, instruction tuning,
future-offset heads, or token-MLP modification. Seed 42. Corpus is pinned by
configs/data/wikipedia.py and configs/data/wikipedia-artifacts.json.

One pass: 600,000,000 stored tokens, context 2048, microbatch 2, accumulation 4,
BF16, AdamW 3e-4, 300 warmup updates then cosine decay. 36,621 updates.

Run from this directory: `.venv/bin/python -m scripts.train`.
Resume: `.venv/bin/python -m scripts.train --resume-latest`.
Results: results/pretraining/wikipedia-en-600m/.

Health checks run in a separate thread every five seconds. GPU >=82 C, CPU
package >=95 C, less than 4 GiB free disk, telemetry failure, SIGINT or SIGTERM
requests a checkpoint at an optimizer boundary and exits. Validation also polls
for stop requests between batches. SIGKILL, power failure, or a failed GPU cannot
guarantee a final save; periodic checkpoints remain available.

Checkpoints every 1,000 updates, and on a health stop or completion, are saved
atomically and mirrored outside the worktree to:
/home/mateus/Documents/Igra-lab-checkpoints/lantern-70m-wikipedia-600m/.
Only the newest two are retained in each location to bound disk usage. Weights
remain excluded from Git; the external backup is the explicit local weight store.
A source/config/tokenizer archive is stored beside backups for recovery.

The host prevents idle sleep during the launched run. See device-health.jsonl
and console.log in the result directory for live status.
