# SMA4 nonlinear contribution experiment — 2026-10-16

Branch model/001-spoony-sma4-gelu16-14m. Read SMA4_GELU16.md and the actual
LatentAttention definition/runtime before changes. Mode sma4_key_gelu16 keeps
independent 4D keys and 4x4 matrices, projects each Mq response through biased
Linear(4,16)+GELU before aggregation, then decodes 16-to-384. Existing cache
format and older attention modes are preserved. Frozen recipe/outputs are in
results/pretraining/sma4-gelu16-14m. Inherited reports describe earlier models.

The fresh 9000-update run is complete; see SMA4_GELU16.md and results/pretraining/sma4-gelu16-14m. All 529 tests and Ruff passed before training.
