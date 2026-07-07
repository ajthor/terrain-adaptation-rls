"""Direct Function Encoder training command.

This is the simple path for day-to-day FE experiments. The generic
``experiments.train`` entrypoint remains available for smoke tests and future
model families, but FE training should not require remembering those flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config
from terrain_adaptation_rls.evaluation.artifacts import write_json

from .common import prepare_run


DEFAULT_CONFIG = "configs/train/warty_fe_scaled.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to a Function Encoder train config. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run-directory name override.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device. Use cuda:N only after checking GPU availability.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for quick debug runs without editing the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config without creating output artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    _require_function_encoder(config, Path(args.config))

    if args.dry_run:
        print(
            f"valid FE train config: {config.name} "
            f"({config.platform}, {config.model.get('n_basis', 8)} bases)"
        )
        return 0

    prepared = prepare_run(
        args.config,
        command="train_fe",
        run_name=args.run_name,
    )
    from terrain_adaptation_rls.training.supervised import run_configured_supervised_training

    metrics = run_configured_supervised_training(
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
