> **Active:** [AC_SWIGLU_30M.md](AC_SWIGLU_30M.md), 14-layer29.705M A16 AC-SwiGLU. Historical notes follow.

Both AC-SwiGLU and matched control training complete. See AC_SWIGLU_COMPARISON.md.575tests/Ruff, shared initial tensor fingerprints and both trained-cache checks passed.

> **Active:** [AC_SWIGLU_14M.md](AC_SWIGLU_14M.md), A16-conditioned SwiGLU and matched pre-FFN-normalized control. Prior notes historical.

# RoPE comparison reruns — 2026-10-16

See ROPE_COMPARISON.md. Active A16/A4/SMA4/key/GELU16 RoPE branches preserve
all prior no-RoPE runs. Positional definitions/runtime/tests have been read;
SMA rotates columns M_i R_i^T, Q rotates R_t q, independent keys rotate R_i k.
Cached rotary values are written once, offsets use count. No Tide experiment
is running; partial tide-a16 branch is preserved. Check results for completion.
Checkpoint links use /var/tmp/igra-rope-14m because home is nearly full.

All five 9000-update RoPE reruns have completed; see ROPE_COMPARISON.md and the per-branch results. Sources were frozen during training.
