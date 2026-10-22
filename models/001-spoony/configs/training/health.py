from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HealthConfig:
    gpu_stop_celsius: int = 82
    cpu_stop_celsius: int = 95
    poll_seconds: float = 5.0
    minimum_free_bytes: int = 4 * 1024**3


health_config = HealthConfig()
checkpoint_backup_directory = Path(
    "/home/mateus/Documents/Igra-lab-checkpoints/lantern-70m-wikipedia-600m"
)
