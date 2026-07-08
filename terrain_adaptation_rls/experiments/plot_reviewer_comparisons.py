"""Create reviewer-facing comparison plots from baseline sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


DEFAULT_SCENE1_RUN = "outputs/eval/20260708T135712Z_reviewer_scene1_all_baselines_mixed"
DEFAULT_SCENE5_RUN = "outputs/eval/20260708T135714Z_reviewer_scene5_all_baselines_mixed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene1-run-dir",
        default=DEFAULT_SCENE1_RUN,
        help="Aggregate baseline sweep directory for the scene1 holdout.",
    )
    parser.add_argument(
        "--scene5-run-dir",
        default=DEFAULT_SCENE5_RUN,
        help="Aggregate baseline sweep directory for the scene5 holdout.",
    )
    parser.add_argument("--run-name", default="reviewer_comparison_plots")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input artifacts without writing plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = {
        "scene1": Path(args.scene1_run_dir),
        "scene5": Path(args.scene5_run_dir),
    }
    for run_dir in run_dirs.values():
        _validate_sweep_dir(run_dir)

    if args.dry_run:
        print(
            "valid reviewer plot inputs: "
            + ", ".join(f"{scene}={path}" for scene, path in run_dirs.items())
        )
        return 0

    artifact_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        artifact_dir / "command.json",
        {
            "command": "plot_reviewer_comparisons",
            "run_dirs": run_dirs,
        },
    )
    from terrain_adaptation_rls.evaluation.reviewer_plots import (
        write_reviewer_comparison_plots,
    )

    write_reviewer_comparison_plots(
        run_dirs=run_dirs,
        artifact_dir=artifact_dir,
    )
    print(artifact_dir)
    return 0


def _validate_sweep_dir(path: Path) -> None:
    for filename in ("method_summary.csv", "window_metrics.csv"):
        if not (path / filename).is_file():
            raise ValueError(f"{path} does not contain {filename}")


if __name__ == "__main__":
    raise SystemExit(main())
