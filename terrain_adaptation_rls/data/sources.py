"""Raw CSV source discovery and manifest helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import io
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CSVSourceSummary:
    """Summary of one source CSV file."""

    path: str
    rows: int
    header: tuple[str, ...]
    first_time: float | None = None
    last_time: float | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["header"] = list(self.header)
        return data


@dataclass(frozen=True)
class SceneSourceSummary:
    """Summary of the raw files available for one scene."""

    platform: str
    scene: str
    files: dict[str, CSVSourceSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "scene": self.scene,
            "files": {name: summary.to_dict() for name, summary in self.files.items()},
        }


def summarize_csv(path: str | Path, *, include_hash: bool = False) -> CSVSourceSummary:
    """Summarize a CSV file without loading it into numeric arrays."""

    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            header = ()
            rows = 0
            first_time = None
            last_time = None
        else:
            rows = 0
            first_time = None
            last_time = None
            for row in reader:
                if not row:
                    continue
                rows += 1
                timestamp = _parse_time(row[0])
                if first_time is None:
                    first_time = timestamp
                last_time = timestamp

    return CSVSourceSummary(
        path=path.as_posix(),
        rows=rows,
        header=header,
        first_time=first_time,
        last_time=last_time,
        sha256=_sha256(path) if include_hash else None,
    )


def discover_scene_sources(
    data_root: str | Path,
    platform: str,
    *,
    include_hash: bool = False,
) -> list[SceneSourceSummary]:
    """Discover scenes under ``data_root/platform``.

    A scene is any directory containing both ``odom.csv`` and ``cmd_vel.csv``.
    The scene name is the POSIX relative path from the platform directory, so
    nested scenes such as ``short_bags/grass`` remain stable.
    """

    platform_root = Path(data_root) / platform
    scenes: list[SceneSourceSummary] = []
    for odom_path in sorted(platform_root.rglob("odom.csv")):
        scene_dir = odom_path.parent
        cmd_vel_path = scene_dir / "cmd_vel.csv"
        if not cmd_vel_path.exists():
            continue
        scene = scene_dir.relative_to(platform_root).as_posix()
        scenes.append(
            summarize_scene_source(
                platform_root,
                platform=platform,
                scene=scene,
                include_hash=include_hash,
            )
        )
    return scenes


def summarize_scene_source(
    platform_root: str | Path,
    *,
    platform: str,
    scene: str,
    include_hash: bool = False,
) -> SceneSourceSummary:
    """Summarize one scene directory."""

    scene_dir = Path(platform_root) / scene
    files = {
        "odom": summarize_csv(scene_dir / "odom.csv", include_hash=include_hash),
        "cmd_vel": summarize_csv(scene_dir / "cmd_vel.csv", include_hash=include_hash),
    }
    trigger_path = scene_dir / "triggers.csv"
    if trigger_path.exists():
        files["triggers"] = summarize_csv(trigger_path, include_hash=include_hash)
    return SceneSourceSummary(platform=platform, scene=scene, files=files)


def write_source_manifest(
    path: str | Path,
    scenes: Iterable[SceneSourceSummary],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a source manifest JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "scenes": [scene.to_dict() for scene in scenes],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_audit_rows(scenes: Iterable[SceneSourceSummary]) -> list[dict[str, Any]]:
    """Convert scene summaries into flat rows for quick audits."""

    rows = []
    for scene in scenes:
        odom = scene.files["odom"]
        cmd_vel = scene.files["cmd_vel"]
        duration = _duration_seconds(odom)
        rows.append(
            {
                "platform": scene.platform,
                "scene": scene.scene,
                "odom_rows": odom.rows,
                "cmd_vel_rows": cmd_vel.rows,
                "duration_seconds": "" if duration is None else duration,
                "has_triggers": "triggers" in scene.files,
            }
        )
    return rows


def source_audit_csv(scenes: Iterable[SceneSourceSummary]) -> str:
    """Format scene summaries as a CSV table."""

    fieldnames = [
        "platform",
        "scene",
        "odom_rows",
        "cmd_vel_rows",
        "duration_seconds",
        "has_triggers",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(source_audit_rows(scenes))
    return output.getvalue()


def _parse_time(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_seconds(summary: CSVSourceSummary) -> float | None:
    if summary.first_time is None or summary.last_time is None:
        return None
    return summary.last_time - summary.first_time
