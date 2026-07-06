"""Streaming evaluation entrypoint skeleton."""

from __future__ import annotations

import argparse

from .common import prepare_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML eval config.")
    parser.add_argument("--run-name", default=None, help="Optional run-directory name override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = prepare_run(
        args.config,
        command="eval_streaming",
        run_name=args.run_name,
    )
    print(prepared.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
