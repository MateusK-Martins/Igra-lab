from src.config.paths import results_path
from src.config.types import CheckpointConfig

checkpoint_config = CheckpointConfig(
    directory=results_path("pretraining/fluxv-no-k-signed-readout/checkpoints"),
    interval=1000,
)
