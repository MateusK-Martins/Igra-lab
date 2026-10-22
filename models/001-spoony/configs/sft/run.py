from pathlib import Path

from src.config.types import RunConfig
from src.sft.config import SFTInitializationConfig

run_config = RunConfig(device="cuda")
initialization_config = SFTInitializationConfig(
    checkpoint_path=Path(
        "/home/mateus/.codex/worktrees/lantern-70m-wikipedia-600m/Igra-lab/models/001-spoony/results/pretraining/wikipedia-en-600m/checkpoints/step-00036621.pt"
    )
)
