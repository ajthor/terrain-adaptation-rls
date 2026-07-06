"""Small artifact helpers for reproducible evaluation outputs."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .k_step import KStepRecord
from .streaming import StreamingRecord


def create_run_dir(
    root: str | Path,
    *,
    run_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Create a timestamped run directory."""

    timestamp = timestamp or datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{stamp}_{run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_streaming_errors_csv(
    path: str | Path,
    records: Iterable[StreamingRecord],
) -> None:
    """Write per-step streaming errors."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "time", "error"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record.index,
                    "time": "" if record.time is None else record.time,
                    "error": record.error,
                }
            )


def write_k_step_errors_csv(
    path: str | Path,
    records: Iterable[KStepRecord],
) -> None:
    """Write per-step k-step rollout errors."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_index",
                "time",
                "step",
                "step_error",
                "accumulated_error",
            ],
        )
        writer.writeheader()
        for record in records:
            accumulated = 0.0
            for step, error in enumerate(record.step_errors, start=1):
                accumulated += error
                writer.writerow(
                    {
                        "window_index": record.index,
                        "time": "" if record.time is None else record.time,
                        "step": step,
                        "step_error": error,
                        "accumulated_error": accumulated,
                    }
                )


def summarize_errors(records: Iterable[StreamingRecord]) -> dict[str, float | int]:
    """Summarize a sequence of streaming records."""

    errors = [float(record.error) for record in records]
    if not errors:
        return {"n": 0, "mean_error": 0.0, "final_accumulated_error": 0.0}
    return {
        "n": len(errors),
        "mean_error": sum(errors) / len(errors),
        "final_accumulated_error": sum(errors),
    }


def summarize_k_step_errors(records: Iterable[KStepRecord]) -> dict[str, float | int]:
    """Summarize k-step rollout records."""

    records = list(records)
    if not records:
        return {
            "n_windows": 0,
            "max_horizon": 0,
            "mean_accumulated_error": 0.0,
        }
    accumulated_errors = [record.accumulated_error for record in records]
    return {
        "n_windows": len(records),
        "max_horizon": max(len(record.step_errors) for record in records),
        "mean_accumulated_error": sum(accumulated_errors) / len(accumulated_errors),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
