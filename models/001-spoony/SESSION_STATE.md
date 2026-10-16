Tide/A16 30M pipeline: model/001-spoony-tide-a16-30m.
Six Tide + three A16/RoPE mixers, width512,FFN1152,26,527,744params.
See TIDE_A16_30M.md. Both stages use6500updates/26,624,000paddedpositions.
733tests/Ruff pass. Existing data/tokenizer/evaluation/training stack unchanged.
Check results/pretraining/tide-a16-30m-v1 and results/sft/tide-a16-30m-qa-v1
for stage completion. Runner: /home/mateus/.codex/experiment-runners/tide-a16-30m.
Large checkpoints link to /var/tmp/igra-tide-a16-30m on the root partition.
Sources must stay frozen during both stages. Push requested after completion.

Both stages are complete. See TIDE_A16_30M.md and per-stage comparison, validation, QA predictions, training curves and probes. Final source/data/settings/exposure/initialization audits passed.
