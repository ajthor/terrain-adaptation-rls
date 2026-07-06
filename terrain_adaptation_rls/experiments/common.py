"""Shared experiment entrypoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config, write_config
from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json
from terrain_adaptation_rls.methods import MethodSpec, validate_method_names


@dataclass(frozen=True)
class PreparedRun:
    """A run directory with its resolved config written."""

    config: ExperimentConfig
    run_dir: Path


@dataclass(frozen=True)
class ValidatedConfig:
    """A config with method names resolved to known method specs."""

    config: ExperimentConfig
    method_specs: tuple[MethodSpec, ...]


def validate_experiment_config(config_path: str | Path) -> ValidatedConfig:
    """Load a config and validate common fields without creating outputs."""

    config = load_config(config_path)
    method_specs = validate_method_names(config.methods) if config.methods else ()
    return ValidatedConfig(config=config, method_specs=method_specs)


def prepare_run(
    config_path: str | Path,
    *,
    command: str,
    run_name: str | None = None,
    timestamp: datetime | None = None,
) -> PreparedRun:
    """Load config and create a run directory for an experiment command."""

    validated = validate_experiment_config(config_path)
    config = validated.config
    method_specs = validated.method_specs
    name = run_name or config.name
    run_root = Path(config.output_root) / config.kind
    run_dir = create_run_dir(run_root, run_name=name, timestamp=timestamp)
    write_config(run_dir / "resolved_config.json", config)
    write_json(
        run_dir / "summary.json",
        {
            "status": "prepared",
            "command": command,
            "name": config.name,
            "kind": config.kind,
            "platform": config.platform,
            "methods": [method.name for method in method_specs],
            "method_specs": [asdict(method) for method in method_specs],
        },
    )
    return PreparedRun(config=config, run_dir=run_dir)
