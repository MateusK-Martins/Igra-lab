from torch.utils.tensorboard import SummaryWriter

from src.config.types import LoggingConfig


def build_writer(
    config: LoggingConfig, *, resume_step: int | None = None
) -> SummaryWriter:
    """Hide abandoned events after the checkpoint's last completed update."""
    return SummaryWriter(
        log_dir=str(config.directory),
        purge_step=None if resume_step is None else resume_step + 1,
    )
