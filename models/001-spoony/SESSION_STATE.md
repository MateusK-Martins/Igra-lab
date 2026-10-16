Completed256-window integrated run: loss2.146345205,PPL8.553539764,9000updates,757tests/Ruff and trained273-token cache checks passed. See GMLP_WINDOW256_14M.md.

> **Active:** [GMLP_WINDOW256_14M.md](GMLP_WINDOW256_14M.md), integrated A16/gMLP with256-token spatial windows. Prior notes historical.

Completed no-Tide integrated/separated comparison. See NO_TIDE_GMLP_COMPARISON.md. Both9000-update runs complete; source/data/settings and trained cache checks passed.

> **Active:** [NO_TIDE_GMLP_14M.md](NO_TIDE_GMLP_14M.md), all-six-layer A16+compressed gMLP, no Tide. Earlier notes are historical.

Completed compressed-output gMLP experiment: final TinyStoriesloss2.023516664/PPL7.564881356,12,652,960parameters,9000updates.754tests+Ruff passed; trained105-token full/cache checks passedFP32/FP64. See LATENT_GMLP_14M.md.

> **Active:** [LATENT_GMLP_14M.md](LATENT_GMLP_14M.md), gMLP after16D attention retrieval, before384D output projection. Prior notes below are historical.

Completed Tide–gMLP–A16: loss1.993348824/PPL7.340073261,9000updates,12,652,160parameters,740tests+Ruff passed. Trained105-token full/cache checks passedFP32/FP64. See GMLP_TIDE_A16.md and results/pretraining/tide-gmlp-a16-14m.

> **Active:** [GMLP_TIDE_A16.md](GMLP_TIDE_A16.md), `[Tide,gMLP,A16/RoPE] ×2`,64-token fixed gMLP windows. Notes below are historical.

Tide/A16 hybrid: four Tide and two single-head 16D Q/K/V attention layers,
RoPE10000, width384/FFN1024, 12,641,280parameters. See TIDE_A16.md.
This experiment is queued after the five compressed-attention RoPE reruns.
Existing Tide/Lantern and no-RoPE branches are preserved. Sources remain frozen
when training starts. Check results/pretraining/tide-a16-14m for completion.

The 9000-update fresh hybrid run is complete. See TIDE_A16.md and results/pretraining/tide-a16-14m. All 730 tests/Ruff and trained cache checks passed.
