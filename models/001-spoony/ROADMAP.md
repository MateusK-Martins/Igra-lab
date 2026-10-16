# Tide template maintenance

The template contains only Tide architecture and shared infrastructure.
No training or new research experiment is part of the cleanup.

Verified: independent sequential recurrence/gradient references, causal and
future-gradient isolation, residual once, full/chunk/single-token cache
equivalence, snapshot/reset/constant-size state, zero/long sequences, BF16
precision/finite gradients, parameter count and ranged initialization.
Shared data/tokenizer/training/checkpoint/SFT and assembly checks remain.

Future models should branch from template/tide, edit configs by concern,
prepare their own artifacts, and record new manifests/results. Preserve
model branches rather than importing live source from another model directory.
