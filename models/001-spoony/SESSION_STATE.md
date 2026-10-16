Tide/A16 hybrid: four Tide and two single-head 16D Q/K/V attention layers,
RoPE10000, width384/FFN1024, 12,641,280parameters. See TIDE_A16.md.
This experiment is queued after the five compressed-attention RoPE reruns.
Existing Tide/Lantern and no-RoPE branches are preserved. Sources remain frozen
when training starts. Check results/pretraining/tide-a16-14m for completion.

The 9000-update fresh hybrid run is complete. See TIDE_A16.md and results/pretraining/tide-a16-14m. All 730 tests/Ruff and trained cache checks passed.
