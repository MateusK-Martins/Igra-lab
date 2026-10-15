# Tide 30M: matched story/QA pipeline

Branch model/001-spoony-tide-30m starts from Tide bd34980; 14M Tide is preserved.
Fresh mixed-corpus training, not resized or warm-started 14M weights.

Architecture: width512, nine layers, SwiGLU1152, tied8192 embedding/head,
global internal learned RMSNorm, sigmoid query, tanh signed candidate,
independent sigmoid retention/write gates. S=g*S_previous+i*tanh(W_V*x).
Existing residual wrapper adds sigmoid(W_Q*x)*RMSNorm(S) once.
Range u_j=j/511, j0..511; g_init=.90+.095*u_j,i_init=.20-.18*u_j.
All gate weights start zero; all feature-wise biases remain learnable.
Original parallel scan/runtime unchanged. FP32 gates/state under BF16.
One512-value FP32 recurrent state/sample/layer (2KiB), no K/matrix/attention.
Exactly29,579,776 parameters, matching original FluxV-30M. Four diagnostic
bins of128 features are not independently normalized heads.

The established30M pipeline is reused without new training infrastructure:
- Stage1: 200000 TinyStories/30000 SQuAD training records, original larger
  training-only8192 tokenizer; context512,batch8,seed42,BF16,AdamW3e-4,
  warmup200,cosinefloor0.1,clip1,6500updates. Existing shuffled ConcatDataset
  and answer-only QA labels. Same split/tokenizer/data hashes and sampler.
- Stage2: initialize only model weights from Tide's own final6500-update
  checkpoint. Fresh optimizer/scheduler; QA-only LR3e-5,warmup100,6500updates,
  batch8,context512,seed42. Same86695 eligible full-passage questions;
  exactly52000 sampled without repetition. Existing scripts.sft unchanged.
- Evaluate final-budget story loss/PPL on1000 stories, QA answer loss/PPL
  on1000 held-out questions, greedy EM/F1 on the same256 questions, max64
  new tokens. Evaluate before/after QA to measure forgetting.
- Each stage consumes26624000 padded input positions. Stage1 actual
  supervision:19748864 story labels +101560 answer/EOS labels.
  Stage2:389671 answer/EOS labels. Exposure replay verifies exact matching.
- Shared30-minute-per-stage protocol retains6500updates; wall time measured.
  Benchmark fit/reliability before full runs; no context/token reduction.

All original30M datasets/tokenizer are reused locally, not regenerated.
QA stage validation file is byte-identical to mixed-stage held-out QA.
Comparison references: Lantern29,565,952 params and FluxV29,579,776 params.
Sources/settings/artifacts/manifests are frozen and independently audited.
PPL cannot be directly compared with14M Tide's different tokenizer/corpus.
Single seed, filtered subsets, weak QA counts: not a leaderboard/reasoning
or general architecture-superiority test. EM/F1 and forgetting both reported.

Verification:703 tests/Ruff pass. Includes scaled full-context forward/
backward, recurrent/cache tests, gate range precision, exact count,
nine independent layers, original pipeline mixed-loader tests, SFT tests,
and evaluator-stage selection. Benchmarks/probes use discarded weights.

Commands from this model directory with its existing .venv:
.venv/bin/python -m scripts.benchmark
.venv/bin/python -m scripts.train
.venv/bin/python -m scripts.validate_comparison
.venv/bin/python -m scripts.sft
.venv/bin/python -m scripts.validate_comparison --sft

New result roots are results/pretraining/tide-30m-v1 and
results/sft/tide-30m-qa-v1. Large datasets/checkpoints remain local/ignored.
Historical experiment documentation below README is retained for provenance.
