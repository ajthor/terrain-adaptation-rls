"""Training entrypoint skeleton."""

from __future__ import annotations

import argparse

from terrain_adaptation_rls.evaluation.artifacts import write_json
from terrain_adaptation_rls.training import DistributedContext

from .common import prepare_run, validate_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML train config.")
    parser.add_argument("--run-name", default=None, help="Optional run-directory name override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and methods without creating output artifacts.",
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Run a tiny synthetic supervised training loop and write metrics.",
    )
    parser.add_argument(
        "--data-train",
        action="store_true",
        help="Train on configured scene data and write metrics, checkpoint, and plots.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for --smoke-train. Defaults to cpu to avoid occupying GPUs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional upper bound on --smoke-train steps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_train and args.data_train:
        raise ValueError("--smoke-train and --data-train are mutually exclusive")

    context = DistributedContext.from_env()
    if args.dry_run:
        validated = validate_experiment_config(args.config)
        if context.is_rank_zero:
            print(
                f"valid train config: {validated.config.name} "
                f"({len(validated.method_specs)} methods)"
            )
        return 0
    prepared = prepare_run(
        args.config,
        command="train",
        run_name=args.run_name,
    )
    write_json(
        prepared.run_dir / "distributed.json",
        {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "is_distributed": context.is_distributed,
        },
    )
    if args.smoke_train:
        from terrain_adaptation_rls.training.supervised import run_synthetic_supervised_training

        metrics = run_synthetic_supervised_training(
            prepared.config,
            device=args.device,
            max_steps=args.max_steps,
        )
        write_json(prepared.run_dir / "training_smoke.json", metrics)
    if args.data_train:
        from terrain_adaptation_rls.training.supervised import run_configured_supervised_training

        metrics = run_configured_supervised_training(
            prepared.config,
            device=args.device,
            max_steps=args.max_steps,
            artifact_dir=prepared.run_dir,
        )
        write_json(prepared.run_dir / "training_metrics.json", metrics)
    if context.is_rank_zero:
        print(prepared.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
