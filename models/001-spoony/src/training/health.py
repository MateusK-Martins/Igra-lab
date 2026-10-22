"""Independent telemetry polling; trainer checkpoints at an update boundary."""

import json
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path


def read_device_health(directory: Path) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    temperature, utilization, memory = map(
        int, result.stdout.strip().splitlines()[0].split(",")
    )
    cpu = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (zone / "type").read_text().strip() in ("x86_pkg_temp", "cpu_thermal"):
                cpu.append(int((zone / "temp").read_text()) / 1000)
        except OSError:
            continue
    return {
        "time": time.time(),
        "gpu_celsius": temperature,
        "cpu_celsius": max(cpu) if cpu else None,
        "gpu_utilization": utilization,
        "gpu_memory_MiB": memory,
        "free_bytes": shutil.disk_usage(directory).free,
    }


def stop_reason(sample: dict, config) -> str | None:
    if sample["gpu_celsius"] >= config.gpu_stop_celsius:
        return f"GPU temperature {sample['gpu_celsius']} C"
    if (
        sample["cpu_celsius"] is not None
        and sample["cpu_celsius"] >= config.cpu_stop_celsius
    ):
        return f"CPU temperature {sample['cpu_celsius']} C"
    if sample["free_bytes"] < config.minimum_free_bytes:
        return "Low disk space"
    return None


class HealthMonitor:
    def __init__(self, directory: Path, config):
        self.directory, self.config = directory, config
        self.reason = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.previous_handlers = {}

    def request_stop(self, reason):
        if self.reason is None:
            self.reason = reason
            print(f"Safe stop requested: {reason}", flush=True)

    def _run(self):
        while not self.closed.is_set():
            try:
                sample = read_device_health(self.directory)
                with (self.directory / "device-health.jsonl").open("a") as file:
                    file.write(json.dumps(sample) + "\n")
                reason = stop_reason(sample, self.config)
                if reason:
                    self.request_stop(reason)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.request_stop(f"Telemetry failure: {error}")
            if self.reason:
                return
            self.closed.wait(self.config.poll_seconds)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[sig] = signal.signal(
                sig, lambda number, frame: self.request_stop(f"Signal {number}")
            )
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.closed.set()
        self.thread.join(timeout=12)
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)
