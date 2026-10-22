# Lantern70M instruction stages — corrected input-only think continuation

Stage 1: 600M-token pretrained model -> 999,849 instruction tokens (9,453 examples).
Stage 2: restart from stage 1 -> 999,995 instruction tokens using the longest 25%
of stage-1 instructions, repeated (2,364 unique examples, 8,970 training records).
Three <think> tokens are supplied as part of the assistant prompt. They receive
no prediction targets: every marker label is masked, and only answer/EOT tokens
are supervised. Token id8192 is a newly added tied vocabulary row, initialized
from the mean of stage-1 embeddings. All existing stage-1 weights are retained.

The earlier marker-prediction experiment is preserved historically under
results/sft/lantern70m-instruct1m-think3/ but is superseded by this corrected run:
results/sft/lantern70m-instruct1m-think3-consume/.
Its REPORT.md and marker-comparison.json compare generation with and without
three supplied markers on stage2_seen, stage1_only, and fully held-out examples.
Run `.venv/bin/python -m scripts.compare_think_consumption` to reproduce all 36
samples, using the tracked evaluation-examples.json and final model export.
Stage 1 validation loss: 2.614870; corrected stage 2 answer-only loss: 2.835003.
The subset/repetition and additional training confound any effect of the markers.

Full checkpoints and inference exports are backed up outside worktrees under
/home/mateus/Documents/Igra-lab-checkpoints/lantern70m-sft1m-{normal,think3-consume}/.
GitHub release lantern70m-instruct1m-think3-consume-v1 stores recovery archives
and SHA256SUMS. Source/config/data snapshots are included with the weights.
