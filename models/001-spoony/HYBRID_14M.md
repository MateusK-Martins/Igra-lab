# Tide/Lantern hybrid 14M

Fresh recipe: [Tide,Tide,Lantern] repeated twice, width384, six layers,
SwiGLU1024, tied8192 embedding/head initstd.02. Four original Tide signed
selective-write/global-RMSNorm mixers and two original Lantern GQA mixers
(8query heads,2KV heads,48features/head, window256, RoPE10000, dropout0).
Exactly13,329,408parameters, below Tide13,774,080 by3.23%; FFN is unchanged.

Outer blocks remain RMSNorm -> residual mixer -> residual SwiGLU -> RMSNorm.
No parallel branch, new router, gate or mixing projection. Tide uses the verified
Triton affine scan for CUDA FP32 accumulation, CPU/FP64 retain the reference scan.
Attention uses the existing SDPA implementation. The two cache types coexist;
the attention history grows with cached tokens under the existing Lantern cache
contract, so the complete hybrid no longer has constant-size inference memory.

Same original14M tokenizer/data hashes, seed42, batch8,context256,BF16,
9000updates,AdamW3e-4,clip1,warmup300/cosinefloor.1,validation500/checkpoints1000.
No warm-start, no data regeneration. Results:results/pretraining/tide-lantern-14m.
Existing benchmark/run_baseline provide timing, validation and fixed generations.
Single seed and different parameter counts limit interpretation. Timing comparisons
with historical models also depend on GPU conditions and scan backend.

## Completed result

9000updates/18,429,440target tokens, validation loss1.983779589, PPL7.270169377.
Training incl.validation/saving:301.300613seconds (5.02min), peak790,150,144bytes
(0.736GiB), throughput61,166target tokens/s. Best saved checkpointstep9000.
All tokenizer/data hashes and training settings match the historical14M pipeline.

Historical Tide:2.010153/PPL7.464462; Lantern:2.018320/PPL7.525675. The hybrid's
loss is lower in this single-seed test, with4Tide/2attention layers and13,329,408
parameters. This does not establish general superiority or retrieval/reasoning
capability. Lantern's exact parameter count is recorded in comparison.json.
Fixed greedy generations and full validation curve are in training-report.json.
Source code is frozen at recipe commitac33073;727tests/Ruff passed before training.

Fixed held-out BF16 probes (recorded without optimizer updates) report finite
outputs and gradients at initialization and after training. Gradient norms:
2.799136 initial,1.744679 final. Full gate/state/output statistics are preserved
in initial-probe.json and final-probe.json. Diagnostic feature bins under global
RMSNorm are not independently normalized or proven semantic heads.
