import subprocess
from dataclasses import replace
from unittest.mock import patch

from configs.training.health import HealthConfig
from src.training.health import HealthMonitor, stop_reason


def test_temperature_disk_and_telemetry_stops(tmp_path):
    config = HealthConfig()
    normal = {"gpu_celsius": 65, "cpu_celsius": 75, "free_bytes": 10 * 1024**3}
    assert stop_reason(normal, config) is None
    assert stop_reason(normal | {"gpu_celsius": 82}, config)
    assert stop_reason(normal | {"cpu_celsius": 95}, config)
    assert stop_reason(normal | {"free_bytes": 0}, config)
    assert stop_reason(normal | {"cpu_celsius": None}, config) is None
    with (
        patch(
            "src.training.health.read_device_health",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ),
        HealthMonitor(tmp_path, replace(config, poll_seconds=0.01)) as monitor,
    ):
        monitor.thread.join(timeout=1)
        assert monitor.reason.startswith("Telemetry failure:")
