"""Lightweight configuration loading and serialization.

The rewrite should not depend on a large configuration framework. This module
keeps configs as plain dictionaries with a small typed envelope for the fields
that every train/eval run should have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


_KNOWN_KEYS = {
    "name",
    "kind",
    "seed",
    "platform",
    "methods",
    "output_root",
    "data",
    "model",
    "training",
    "evaluation",
    "metadata",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Common config envelope for train/eval commands."""

    name: str
    kind: str
    seed: int = 0
    platform: str | None = None
    methods: tuple[str, ...] = ()
    output_root: str = "outputs"
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config from a dictionary-like object."""

        if "name" not in mapping:
            raise ValueError("config is missing required field 'name'")
        if "kind" not in mapping:
            raise ValueError("config is missing required field 'kind'")

        extras = {key: value for key, value in mapping.items() if key not in _KNOWN_KEYS}
        methods = mapping.get("methods", ())
        if isinstance(methods, str):
            methods = (methods,)
        return cls(
            name=str(mapping["name"]),
            kind=str(mapping["kind"]),
            seed=int(mapping.get("seed", 0)),
            platform=_optional_str(mapping.get("platform")),
            methods=tuple(str(method) for method in methods),
            output_root=str(mapping.get("output_root", "outputs")),
            data=dict(mapping.get("data", {})),
            model=dict(mapping.get("model", {})),
            training=dict(mapping.get("training", {})),
            evaluation=dict(mapping.get("evaluation", {})),
            metadata=dict(mapping.get("metadata", {})),
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON/YAML output."""

        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "seed": self.seed,
            "output_root": self.output_root,
            "methods": list(self.methods),
            "data": self.data,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
            "metadata": self.metadata,
        }
        if self.platform is not None:
            data["platform"] = self.platform
        data.update(self.extras)
        return data


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a JSON or YAML config file."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(path)
    else:
        raise ValueError(f"Unsupported config file suffix: {path.suffix}")
    if not isinstance(payload, Mapping):
        raise ValueError("config file must contain a mapping at the top level")
    return ExperimentConfig.from_mapping(payload)


def write_config(path: str | Path, config: ExperimentConfig) -> None:
    """Write a resolved config as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YAML configs require PyYAML. Use JSON on minimal hosts or install PyYAML "
            "inside the devcontainer."
        ) from exc
    return yaml.safe_load(path.read_text())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
