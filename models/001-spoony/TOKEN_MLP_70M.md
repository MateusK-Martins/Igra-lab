# Lantern 70M with full shared TokenMLP: equal exposure

Branch: `codex/lantern-70m-token-mlp-shared64`, based on the completed baseline
`codex/lantern-70m-wikipedia`. Earlier experiment branches/worktrees are preserved.

All nine FFNs are replaced with residual TokenMLP. Keep width 768, nine layers,
eight query heads, two KV heads, head width 96, unbounded causal attention,
RoPE base 10000, context 2048, RMSNorm epsilon 1e-6 and tied input/head weights.
Input embedding initialization retains baseline std 0.016329931618554522.
Two token tables of width 64 and two bias-free 768 -> 64 -> 768 projections
are shared across all nine branches; token table initialization std stays 0.02.

    output = x + W_up (tanh(A[token_id] B[token_id]^T) W_down x)

No parameter compensation changes are made elsewhere. CUDA TokenMLP uses the
previously verified fused formula. The 70M baseline's original attention runtime
is preserved, including its existing causal SDPA path.

| Component | Unique parameters at vocabulary 8192 |
| --- | ---: |
| Input embedding / tied head | 6,291,456 |
| Two shared token tables | 1,048,576 |
| Shared down/up projections | 98,304 |
| Nine original GQA branches | 13,271,040 |
| Eighteen RMSNorms | 13,824 |
| Total | 20,723,200 |

## Equal comparison protocol

The user explicitly selected equal training exposure over the earlier ten-minute
limit. Use the completed original 100M-token Wikipedia run as reference, not the
older 600M-token plan in historical documentation. All pinned prepared-artifact
hashes match `configs/data/wikipedia-100m-artifacts.json`, including the canonical
8192-token vocabulary and the separate 2M-token validation store.

One full pass: 6,104 optimizer updates, 24,414 microbatches, 99,999,744 supervised
target positions (trailing stored tokens without a full window are unused).
Original microbatch 2, accumulation 4, effective batch 8 sequences, context and
stride 2048, seed 42, BF16, AdamW LR 3e-4, betas (0.9,0.95), epsilon 1e-8,
decay 0.01, clipping 1.0, 300 warmup updates and cosine floor 0.1 are preserved.
Validate every 500 updates, save every 1,000 and at completion. No QA mixing.

Original baseline: 70,006,272 parameters, validation loss 3.1807894921693647,
perplexity 24.06574577181039 after the same 100M-token pass; recorded training
duration is approximately 85 minutes (not an exact instrumented timing).

Fresh outputs: `results/pretraining/lantern70-tokenmlp-shared64-wikipedia100m-v1/`.
Readable logs/reports are mirrored into the main workspace at the same relative
results path. Run with `.venv/bin/python -m scripts.run_experiment`; it invokes
the original training pipeline, preserves export/checkpoint behavior and records
final timing, metrics and a strict checkpoint restoration check. There is no
shortened schedule or ten-minute process timeout for this equal-exposure run.

## Verification and benchmark

507 tests pass, including configured model gradients, sharing/parameter counts,
cache equivalence, checkpoint restoration and CUDA FP32/BF16 reference gradients.
Initialization validation compares the preserved scaled embedding with N(0,1),
rather than imposing the smaller model's arbitrary two-nat initial-loss bound.
Ruff checks and formatting pass. Real-data benchmark: 0.51439 seconds/update,
31,851 tokens/s, full validation 17.34 seconds, peak allocated CUDA memory
1,728,516,096 bytes. Full-size weight save/load logits matched exactly.
Benchmark weights are discarded. A microbatch-4/accumulation-2 experiment was
slower (0.58967 seconds/update), so original microbatch/accumulation are retained.

## Completed equal-exposure comparison

The fresh TokenMLP run completed all 6,104 updates and one full data pass:
99,999,744 target tokens, 24,414 microbatches, seed 42. The training report
measures 3,364.19 seconds (56.1 minutes), including loading, validation,
checkpoint saves and export, with peak allocated VRAM 1,728,516,096 bytes.
Final and best saved validation loss are both 3.4964073 (perplexity 32.99669)
at update 6,104. The original baseline measured 3.1807895 (perplexity 24.06575)
at equal data exposure and has 70,006,272 parameters; TokenMLP has 20,723,200.
Thus TokenMLP uses 70.4% fewer parameters and has 0.31562 higher validation
cross-entropy nats on this single matched-seed run. The original baseline time
is recorded only approximately (~85 minutes), so a precise speedup comparison
is unavailable.

The final checkpoint and exported `model.pt` are saved under the run directory.
The checkpoint state records 6,104 updates, 24,414 batches, 99,999,744 tokens,
one completed epoch and validation loss 3.4964073. Checkpoints exist at updates
1,000 through 6,000 and at the final update 6,104. The model manifest records
the pinned data hashes, original context/batch/optimizer/scheduler settings,
the current code snapshot and GPU environment. The JSON training report
contains the validation curve, final metrics and measured run time.
