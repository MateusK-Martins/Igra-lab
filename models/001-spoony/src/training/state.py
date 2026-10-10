from dataclasses import dataclass


@dataclass
class TrainingState:
    updates: int = 0
    batches: int = 0
    tokens: int = 0
    completed_epochs: int = 0
    last_loss: float | None = None
