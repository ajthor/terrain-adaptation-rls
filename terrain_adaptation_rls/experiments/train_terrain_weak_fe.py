"""Train a pure weak-form Function Encoder on terrain data."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config
from terrain_adaptation_rls.evaluation.artifacts import write_json

from .common import prepare_run


DEFAULT_CONFIG = "configs/train/small_ugv_fe_weak_ice_holdout.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    _require_function_encoder(config, Path(args.config))
    if args.dry_run:
        print(
            f"valid pure weak FE train config: {config.name} "
            f"({config.platform}, {config.model.get('n_basis', 16)} bases)"
        )
        return 0

    prepared = prepare_run(
        args.config,
        command="train_terrain_weak_fe",
        run_name=args.run_name,
    )
    from terrain_adaptation_rls.training.terrain_weak_fe import run_terrain_weak_fe_training

    metrics = run_terrain_weak_fe_training(
        prepared.config,
        device=args.device,
        max_steps=args.max_steps,
        artifact_dir=prepared.run_dir,
    )
    write_json(prepared.run_dir / "training_metrics.json", metrics)
    print(prepared.run_dir)
    return 0


def _require_function_encoder(config: ExperimentConfig, path: Path) -> None:
    family = str(config.model.get("family", ""))
    if family not in {"function_encoder", "fe"}:
        raise ValueError(
            f"{path} is not a Function Encoder config; "
            f"model.family is '{family or '<missing>'}'"
        )


if __name__ == "__main__":
    raise SystemExit(main())
