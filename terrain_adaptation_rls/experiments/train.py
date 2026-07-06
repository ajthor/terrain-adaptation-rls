"""Training entrypoint skeleton."""

from __future__ import annotations

import argparse

from terrain_adaptation_rls.evaluation.artifacts import write_json
from terrain_adaptation_rls.training import DistributedContext

from .common import prepare_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML train config.")
    parser.add_argument("--run-name", default=None, help="Optional run-directory name override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = DistributedContext.from_env()
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
    if context.is_rank_zero:
        print(prepared.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
