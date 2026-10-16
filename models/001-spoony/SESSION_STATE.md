# SMA4 independent-key experiment — 2026-10-16

Branch model/001-spoony-sma4-key-14m. See SMA4_KEY.md for the active mechanism
and frozen protocol. Original A16/A4/SMA4 branches are preserved. Definition
LatentAttention mode sma4_key builds the same full-width backbone, independent
4D score keys, 4x4 matrix responses, and a keyed matrix cache with snapshot
format 3. Existing modes and old snapshots remain supported. Read the runtime,
definition and tests before further edits. Check results/pretraining/sma4-key-14m
for run completion; no historical SMA4 report describes this new run.

Training is now complete: 9000 updates / 18,429,440 target tokens. Reports and probes are in results/pretraining/sma4-key-14m. No runtime changes were made during the frozen run.
