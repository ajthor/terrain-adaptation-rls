"""Print a quick CSV audit of raw data sources."""

from __future__ import annotations

import argparse
import sys

from terrain_adaptation_rls.data.sources import discover_scene_sources, source_audit_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="terrain_adaptation_rls/data",
        help="Root directory containing platform subdirectories.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        required=True,
        help="Platform to audit. May be passed multiple times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenes = []
    for platform in args.platform:
        scenes.extend(discover_scene_sources(args.data_root, platform))
    sys.stdout.write(source_audit_csv(scenes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
