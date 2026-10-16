TDM attention hybrid completed: branch model/001-spoony-tdm-hybrid-14m.
Four pre-norm attention -> TDM -> SwiGLU layers, width384,FFN1024,13,481,856params.
TDM scales1,2,4,8,16,32; zero-padded causal adjacent-window averages, full-width W_m,
uniform-initialized learned softmax routing and W_o. No Tide recurrence/gates.
Typed HistoryEntry stores63past inputs; snapshotformat2 (oldKV/stateformat1supported).
751tests/Ruff passed.9000updates/18,429,440tokens completed:loss1.996956/PPL7.366597,
time493.290s, peak841,324,032bytes. Sources/settings/data fingerprints verified.
Compared with previous hybrid1.983780, TDM version is slower and has slightly higher
loss; not an isolated ablation due depth/attention/FFN/norm differences. Scale1mean
routing~.2812; no semantic specialization claim. Initial/final BF16 probes finite.
See TDM_14M.md and results/pretraining/tdm-hybrid-14m/. No trainer/inhibitor remains.
No extra control run, QA stage or30M TDM requested/launched. Data/checkpoints ignored;
reports saved. No push requested.
