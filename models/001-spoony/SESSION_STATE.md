# Completed compressed-attention experiment — 2026-10-16

Active branch: model/001-spoony-a16-14m.
Read ATTENTION_COMPARISON.md, configs/model/model.py, src/model/latent_attention.py,
and src/model/cache_storage.py before further work. Independent A16/A4/SMA4 runs
are complete. Each used 9000 fresh updates, the existing Lantern/TinyStories
pipeline, six width-384 layers and SwiGLU 1024. All three passed 507 tests/Ruff.
Large checkpoints and prepared data remain ignored locally. No trainer remains
active after the sequential suite exits. Branches are independent; no live shared
source dependency. Existing model worktrees and user checkout were preserved.
No pushes were requested for this experiment.
