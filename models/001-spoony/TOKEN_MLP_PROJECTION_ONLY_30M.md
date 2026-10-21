# Lantern 30M projection-only TokenMLP ablation

Branch: `codex/lantern-30m-tokenmlp-projection-only`, based on
`codex/lantern-30m-token-mlp-shared64`.

This matches the full 30M TokenMLP experiment's nine-layer architecture and
training recipe. The only model change removes the two shared token embedding
tables and the token-conditioned `tanh(a \otimes b)` matrix operation. The
shared bottleneck projections remain:

```text
z = W_down x       # 512 -> 64
output = x + W_up z # 64 -> 512, with the declared residual
```

Attention, norms, input embedding and tied LM head, tokenizer, prepared story
and QA data, initialization seed, optimizer, scheduler, data order, precision,
batch size, update count and validation cadence match the full TokenMLP run.
The attention and Transformer dimensions are unchanged. No parameters are added
to compensate for the removed token matrices.

At vocabulary 8192, the expected parameter budget is:

| Component | Unique parameters |
| --- | ---: |
| Input embedding / tied head | 4,194,304 |
| Shared down/up projections | 65,536 |
| Nine original GQA branches | 5,898,240 |
| Eighteen RMSNorms | 9,216 |
| Total | 10,167,296 |

The full shared TokenMLP model has 11,215,872 parameters; original Lantern 30M
has 29,565,952. The experimental run writes to
`results/pretraining/lantern-30m-tokenmlp-projection-only-v1/`, keeping all
baseline outputs intact.

## Matched protocol

Train for 6,500 updates on the identical seeded story/QA `ConcatDataset`,
processing 26,624,000 padded input positions with batch 8, context/stride 512,
BF16, seed 42, AdamW learning rate 3e-4, 200 warmup updates, cosine decay to
0.1, and validation every 1,000 updates. Checkpoints are saved every 2,000
updates and at completion. This matches the full TokenMLP and original Lantern
comparison protocol; this remains a single-seed result.

## Results

The run completed all **6,500 updates in 414.78 seconds (6m 55s)**, consuming
**26,624,000 padded input positions**. Final story validation loss was **2.371041**
(perplexity **10.708539**); held-out answer-only QA loss was **6.331085**
(perplexity **561.765934**). Greedy QA generation over the same 256 examples
scored **0.00 EM / 17.13 F1**. Peak allocated VRAM was 1,159,946,240 bytes.
The final checkpoint is `checkpoints/step-00006500.pt` in the result directory;
checkpoints remain local and ignored.

| Model | Parameters | Story loss / PPL | QA answer loss / PPL | QA EM / F1 (%) | Training time |
| --- | ---: | --- | --- | --- | ---: |
| Original Lantern 30M | 29,565,952 | 2.055921 / 7.814034 | 6.080587 / 437.285769 | 0.00 / 1.19 | 1,228.95 s |
| Full shared TokenMLP 64 | 11,215,872 | 2.284905 / 9.824748 | 6.261028 / 523.757186 | Not evaluated | 473.84 s |
| Projection-only TokenMLP 64 | 10,167,296 | 2.371041 / 10.708539 | 6.331085 / 561.765934 | 0.00 / 17.13 | 414.78 s |

Against full TokenMLP, removing the token tables and matrix reduced parameters by
1,048,576 (**9.35%**) and training time by 59.06 seconds (**12.5%**). Story loss
rose by 0.086137 and QA answer loss by 0.070057. The two TokenMLP runs have
identical prepared-data hashes and identical resolved training settings in their
manifests. The full TokenMLP run did not evaluate generation, so its EM/F1 is not
directly available for comparison. These are single-seed measurements; the
projection-only generation score should not be interpreted as a reliable quality
gain from one evaluation sample set.

The result directory contains the benchmark, manifest, training report,
held-out QA predictions, and validation report. Samples from TinyStories
generation are in `training-report.json`.
