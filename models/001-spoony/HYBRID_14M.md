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
