"""Virtual split policies and manifests.

The rewrite keeps processed scene data together and records train/validation/test
membership as policy metadata. Index files can be materialized later for frozen
paper runs, but they are not required for day-to-day experimentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SceneSplitPolicy:
    """Scene-level train/validation/test split definition."""

    name: str
    platform: str
    seed: int
    train_scenes: tuple[str, ...]
    validation_scenes: tuple[str, ...] = ()
    test_scenes: tuple[str, ...] = ()
    version: str = "1"
    description: str = ""

    def manifest(
        self,
        *,
        preprocessing_version: str,
        source_metadata: Mapping[str, Any] | None = None,
        materialized_indices: bool = False,
    ) -> "SplitManifest":
        return SplitManifest(
            name=self.name,
            platform=self.platform,
            seed=self.seed,
            policy_name="scene_split",
            policy_version=self.version,
            preprocessing_version=preprocessing_version,
            train_scenes=self.train_scenes,
            validation_scenes=self.validation_scenes,
            test_scenes=self.test_scenes,
            materialized_indices=materialized_indices,
            description=self.description,
            source_metadata=dict(source_metadata or {}),
        )


@dataclass(frozen=True)
class SplitManifest:
    """Serializable metadata for a split policy."""

    name: str
    platform: str
    seed: int
    policy_name: str
    policy_version: str
    preprocessing_version: str
    train_scenes: tuple[str, ...]
    validation_scenes: tuple[str, ...] = ()
    test_scenes: tuple[str, ...] = ()
    materialized_indices: bool = False
    description: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["train_scenes"] = list(self.train_scenes)
        data["validation_scenes"] = list(self.validation_scenes)
        data["test_scenes"] = list(self.test_scenes)
        return data

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def split_indices(
    n_items: int,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, list[int]]:
    """Create deterministic train/validation/test indices.

    Args:
        n_items: Number of items in the scene.
        seed: RNG seed for shuffling indices.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.

    Returns:
        Dictionary with ``train``, ``validation``, and ``test`` index lists.
    """

    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    _validate_fraction("train_fraction", train_fraction)
    _validate_fraction("validation_fraction", validation_fraction)
    if train_fraction + validation_fraction > 1.0:
        raise ValueError("train_fraction + validation_fraction must be <= 1")

    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n_items * train_fraction)
    n_validation = int(n_items * validation_fraction)
    train_end = n_train
    validation_end = train_end + n_validation

    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def scene_source_metadata(row_counts: Mapping[str, int]) -> dict[str, dict[str, int]]:
    """Build simple source metadata from scene row counts."""

    return {scene: {"rows": int(rows)} for scene, rows in row_counts.items()}


def _validate_fraction(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
