"""Short measurement only; benchmark weights are discarded."""

WARMUP_UPDATES = 3
MEASURED_UPDATES = 10

# None measures the training defaults; smaller values preserve effective batch.
MICRO_BATCH_SIZE = None
