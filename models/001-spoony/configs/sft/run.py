from src.config.paths import results_path
from src.config.types import RunConfig
from src.sft.config import SFTInitializationConfig

run_config = RunConfig(device="cuda")
initialization_config = SFTInitializationConfig(
    checkpoint_path=results_path(
        "pretraining/comparison-fluxv-30m-v2/checkpoints/step-00006500.pt"
    )
)
