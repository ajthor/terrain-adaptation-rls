"""Generate a raw data source manifest."""

from __future__ import annotations

import argparse

from terrain_adaptation_rls.data.sources import discover_scene_sources, write_source_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="terrain_adaptation_rls/data",
        help="Root directory containing platform subdirectories.",
    )
    parser.add_argument("--platform", required=True, help="Platform name under data root.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument(
        "--include-hash",
        action="store_true",
        help="Include SHA-256 hashes. This is slower on large CSVs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenes = discover_scene_sources(
        args.data_root,
        args.platform,
        include_hash=args.include_hash,
    )
    write_source_manifest(
        args.output,
        scenes,
        metadata={
            "data_root": args.data_root,
            "platform": args.platform,
            "include_hash": args.include_hash,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
