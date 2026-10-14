from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class TrainingState:
    updates: int = 0
    batches: int = 0
    tokens: int = 0
    supervised_tokens: int = field(default=0, kw_only=True)
    completed_epochs: int = 0
    last_loss: float | None = None
    last_validation_loss: float | None = None
    last_validation_perplexity: float | None = None
    batches_in_epoch: int = 0
    epoch_generator_state: Tensor | None = field(
        default=None, repr=False, compare=False
    )
