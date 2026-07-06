"""Helpers for checking GPU availability before launching jobs."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class GPUStatus:
    """Summary of one GPU reported by nvidia-smi."""

    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


def query_gpus() -> list[GPUStatus]:
    """Query GPU status using nvidia-smi.

    Returns an empty list if nvidia-smi is unavailable or cannot communicate
    with a driver. This makes launcher code conservative on machines without
    visible GPUs.
    """

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    return parse_nvidia_smi_csv(result.stdout)


def parse_nvidia_smi_csv(output: str) -> list[GPUStatus]:
    """Parse the CSV output from ``query_gpus``."""

    statuses: list[GPUStatus] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Expected 5 CSV fields from nvidia-smi, got {len(parts)}: {line}")
        index, name, memory_used, memory_total, utilization = parts
        statuses.append(
            GPUStatus(
                index=int(index),
                name=name,
                memory_used_mb=int(memory_used),
                memory_total_mb=int(memory_total),
                utilization_percent=int(utilization),
            )
        )
    return statuses


def available_gpus(
    statuses: list[GPUStatus],
    *,
    max_memory_used_mb: int = 500,
    max_utilization_percent: int = 10,
) -> list[int]:
    """Return GPU indices that look idle enough for a new run."""

    return [
        status.index
        for status in statuses
        if status.memory_used_mb <= max_memory_used_mb
        and status.utilization_percent <= max_utilization_percent
    ]
