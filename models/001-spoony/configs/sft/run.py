from src.config.paths import results_path
from src.config.types import RunConfig
from src.sft.config import SFTInitializationConfig

run_config = RunConfig(device="cuda")
initialization_config = SFTInitializationConfig(
    checkpoint_path=results_path(
        "pretraining/shakespeare-v4/checkpoints/step-00002400.pt"
    )
)
