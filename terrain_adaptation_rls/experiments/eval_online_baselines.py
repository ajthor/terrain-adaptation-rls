"""Compare online terrain-adaptation baselines on one scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fe-run-dir",
        required=True,
        help="FE training artifact directory with resolved_config.json and function_encoder_model.pth.",
    )
    parser.add_argument(
        "--neuralfly-run-dir",
        default=None,
        help="Optional NeuralFly-style training artifact directory.",
    )
    parser.add_argument("--scene", default="scene1", help="Scene to stream through each method.")
    parser.add_argument("--run-name", default="online_baselines", help="Optional run name.")
    parser.add_argument("--output-root", default="outputs", help="Root for eval artifacts.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Evaluation device. Use cuda:N only after checking GPU availability.",
    )
    parser.add_argument("--max-points", type=int, default=512, help="Maximum streaming points.")
    parser.add_argument("--start-index", type=int, default=0, help="First scene index to stream.")
    parser.add_argument(
        "--n-example-points",
        type=int,
        default=None,
        help="Number of same-scene points for offline coefficient solves.",
    )
    parser.add_argument("--forgetting-factor", type=float, default=0.95)
    parser.add_argument("--initial-covariance", type=float, default=1000.0)
    parser.add_argument("--measurement-noise", type=float, default=1e-6)
    parser.add_argument(
        "--linear-no-bias",
        action="store_true",
        help="Disable the constant feature in the no-training linear baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifact directories and arguments without creating artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fe_run_dir = Path(args.fe_run_dir)
    neuralfly_run_dir = None if args.neuralfly_run_dir is None else Path(args.neuralfly_run_dir)
    _validate_fe_run_dir(fe_run_dir)
    if neuralfly_run_dir is not None:
        _validate_neuralfly_run_dir(neuralfly_run_dir)

    if args.dry_run:
        neuralfly_text = (
            "none" if neuralfly_run_dir is None else neuralfly_run_dir.as_posix()
        )
        print(
            f"valid online baseline eval: fe_run={fe_run_dir} "
            f"neuralfly_run={neuralfly_text} scene={args.scene}"
        )
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        run_dir / "command.json",
        {
            "command": "eval_online_baselines",
            "fe_run_dir": fe_run_dir,
            "neuralfly_run_dir": neuralfly_run_dir,
            "scene": args.scene,
            "device": args.device,
            "max_points": args.max_points,
            "start_index": args.start_index,
            "n_example_points": args.n_example_points,
            "forgetting_factor": args.forgetting_factor,
            "initial_covariance": args.initial_covariance,
            "measurement_noise": args.measurement_noise,
            "linear_include_bias": not args.linear_no_bias,
        },
    )

    from terrain_adaptation_rls.evaluation.online_baselines import (
        run_online_baseline_comparison,
    )

    run_online_baseline_comparison(
        fe_run_dir=fe_run_dir,
        neuralfly_run_dir=neuralfly_run_dir,
        artifact_dir=run_dir,
        scene=args.scene,
        device=args.device,
        max_points=args.max_points,
        start_index=args.start_index,
        n_example_points=args.n_example_points,
        forgetting_factor=args.forgetting_factor,
        initial_covariance=args.initial_covariance,
        measurement_noise=args.measurement_noise,
        linear_include_bias=not args.linear_no_bias,
    )
    print(run_dir)
    return 0


def _validate_fe_run_dir(path: Path) -> None:
    if not (path / "resolved_config.json").is_file():
        raise ValueError(f"{path} does not contain resolved_config.json")
    if not (path / "function_encoder_model.pth").is_file():
        raise ValueError(f"{path} does not contain function_encoder_model.pth")


def _validate_neuralfly_run_dir(path: Path) -> None:
    if not (path / "resolved_config.json").is_file():
        raise ValueError(f"{path} does not contain resolved_config.json")
    if not (path / "neuralfly_style_basis.pth").is_file():
        raise ValueError(f"{path} does not contain neuralfly_style_basis.pth")


if __name__ == "__main__":
    raise SystemExit(main())
