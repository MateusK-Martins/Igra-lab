# Lantern30M — Llama-style decoder, vocabulary 32K

Branch: codex/lantern30m-llama-vocab32k. Derived from the original Lantern70M
architecture branch codex/lantern-70m-wikipedia. This is an architecture/tokenizer
branch: no full model training has been launched, and no trained weights are
being represented as resized pretrained weights. The CUDA verification weights
were discarded. Original 70M checkpoints are shape/tokenizer incompatible.

| Setting | Value |
|---|---:|
| Parameters | 29,890,048 |
| Decoder layers | 5 |
| Hidden width | 512 |
| SwiGLU intermediate width | 1,280 |
| Vocabulary | 32,768 |
| Training context / stride | 2,048 / 2,048 |
| Query / KV heads | 8 / 2 |
| Head dimension | 64 |
| RoPE base | 10,000 |
| RMSNorm epsilon | 1e-6 |
| Dropout / projection biases | 0 / none |
| Input embedding / output head | tied |

Each layer computes:

```text
h = x + GQA(RMSNorm(x))
y = h + SwiGLU(RMSNorm(h))
```

A final RMSNorm precedes the tied linear vocabulary projection. This is the
Llama-style pre-norm decoder organization. Retaining tied embeddings is an
explicit small-model design choice; this is not a byte-for-byte Meta Llama
configuration. The old Lantern branch normalized outside the residual branches
and placed no separate normalization between attention and the MLP.

Parameter breakdown: shared embedding/head 16,777,216; attention 3,276,800;
SwiGLU 9,830,400; RMSNorm weights 5,632. The larger vocabulary occupies 56.1%
of the parameter budget. BF16/FP16 weight storage is approximately 57.0 MiB,
excluding optimizer/gradients/activations. Full 2,048-token KV cache, batch1,
BF16: 5.0 MiB. Attention remains causal and unbounded, as before; 2,048 is the
configured training context, not a new runtime cap or a long-context claim.

The new byte-level BPE tokenizer was trained on the first 20,000 deduplicated
Wikipedia training documents, excluding validation data. It has exactly 32,768
tokens and retains <|unk|>/<|eot|>. It is saved under configs/tokenizers/.
See results/tokenizer/wikipedia-en-32768/report.json for source and hashes.
The old 8K token stores cannot be reused; retokenize before model training.
Run python -m scripts.prepare_wikipedia to prepare/pin the new corpus. Default
training outputs have distinct names to protect existing experiment results.

494 tests pass, including pre-norm residual identity/equations, finite gradients,
serialization/tied weights, cache isolation and cached-vs-full forward parity.
The actual 32K model also completed a BF16 CUDA optimizer update at batch2 and
context2048 with finite gradients. See the machine-readable model-report.json
under results/architecture/lantern30m-llama-vocab32k/ for measured peak memory.

## Active full training: 600M tokens

Prepare an exact 600,000,000-token training slice plus 2,000,000 held-out tokens
with the frozen 32K tokenizer. This is a new token budget; old 8K token stores
are not reused. Train from fresh random weights with seed42, BF16, context2048,
microbatch2/accumulation4. 36,621 updates, AdamW3e-4, 300 warmup steps, cosine
floor0.1, validation every500, checkpoints every1000 and at completion.
Outputs: results/pretraining/lantern30m-llama-vocab32k-wikipedia600m/.
Benchmark: 22.136 seconds/million actual new-tokenizer Wikipedia tokens;
45,175 tokens/s; 2.573 GiB peak allocation. Estimated full run: 4-4.5 hours
including validation/checkpoint overhead, excluding corpus preparation.

Health telemetry every5 seconds: GPU >=82 C, CPU >=95 C, telemetry failure,
low disk (<4 GiB), SIGINT/SIGTERM requests an atomic save and stop at an
optimizer boundary; validation can also be interrupted safely. Periodic and
final full-state checkpoints are mirrored to:
/home/mateus/Documents/Igra-lab-checkpoints/lantern30m-llama-vocab32k-wikipedia600m/.
Keep newest two in each location. Weights are not stored by a normal Git push;
this external mirror preserves local recovery, and a release upload is needed
for remote weight storage. A frozen source/config/tokenizer archive is saved
beside the backup before training. Resume with --resume-latest, preserving
source/config/data identities. Idle sleep is inhibited during this run.

Validation loss/perplexity are expressed in the new 32K tokenization and should
not be directly compared with the old 8K-tokenizer model.
