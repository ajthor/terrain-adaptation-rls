"""Streaming evaluation entrypoint skeleton."""

from __future__ import annotations

import argparse

from .common import prepare_run, validate_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML eval config.")
    parser.add_argument("--run-name", default=None, help="Optional run-directory name override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and methods without creating output artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        validated = validate_experiment_config(args.config)
        print(
            f"valid eval config: {validated.config.name} "
            f"({len(validated.method_specs)} methods)"
        )
        return 0
    prepared = prepare_run(
        args.config,
        command="eval_streaming",
        run_name=args.run_name,
    )
    print(prepared.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
