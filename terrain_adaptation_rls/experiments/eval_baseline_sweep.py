"""Run reviewer-style aggregate baseline evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.configuration import load_config
from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json

from .eval_online_baselines import (
    _validate_fe_run_dir,
    _validate_neuralfly_run_dir,
    _validate_node_run_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fe-run-dir",
        required=True,
        help="FE training artifact directory with resolved_config.json and function_encoder_model.pth.",
    )
    parser.add_argument("--neuralfly-run-dir", default=None)
    parser.add_argument("--node-run-dir", default=None)
    parser.add_argument(
        "--split",
        default="heldout",
        choices=("validation", "test", "heldout", "train", "all_config"),
        help="Config split to evaluate when --scenes is not provided.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Explicit scene names to evaluate, overriding --split.",
    )
    parser.add_argument("--run-name", default="baseline_sweep")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=1)
    parser.add_argument("--n-example-points", type=int, default=None)
    parser.add_argument("--forgetting-factor", type=float, default=0.95)
    parser.add_argument("--initial-covariance", type=float, default=1000.0)
    parser.add_argument("--measurement-noise", type=float, default=1e-6)
    parser.add_argument("--skip-fe-variants", action="store_true")
    parser.add_argument("--kalman-process-noise", type=float, default=0.0)
    parser.add_argument("--fe-sgd-learning-rate", type=float, default=1.0)
    parser.add_argument("--fe-sgd-momentum", type=float, default=0.0)
    parser.add_argument("--fe-sgd-weight-decay", type=float, default=0.0)
    parser.add_argument("--fe-window-size", type=int, default=100)
    parser.add_argument("--fe-window-ridge", type=float, default=1e-6)
    parser.add_argument("--linear-no-bias", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print resolved scenes/windows without creating artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fe_run_dir = Path(args.fe_run_dir)
    neuralfly_run_dir = None if args.neuralfly_run_dir is None else Path(args.neuralfly_run_dir)
    node_run_dir = None if args.node_run_dir is None else Path(args.node_run_dir)
    _validate_fe_run_dir(fe_run_dir)
    if neuralfly_run_dir is not None:
        _validate_neuralfly_run_dir(neuralfly_run_dir)
    if node_run_dir is not None:
        _validate_node_run_dir(node_run_dir)

    from terrain_adaptation_rls.evaluation.baseline_sweep import resolve_sweep_windows

    config = load_config(fe_run_dir / "resolved_config.json")
    windows = resolve_sweep_windows(
        config,
        scenes=args.scenes,
        split=args.split,
        max_points=args.max_points,
        window_stride=args.window_stride,
        max_windows_per_scene=args.max_windows_per_scene,
    )

    if args.dry_run:
        window_text = ", ".join(
            f"{window.split}:{window.scene}@{window.start_index}" for window in windows
        )
        print(f"valid baseline sweep: {len(windows)} windows [{window_text}]")
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        run_dir / "command.json",
        {
            "command": "eval_baseline_sweep",
            "fe_run_dir": fe_run_dir,
            "neuralfly_run_dir": neuralfly_run_dir,
            "node_run_dir": node_run_dir,
            "split": args.split,
            "scenes": args.scenes,
            "device": args.device,
            "max_points": args.max_points,
            "window_stride": args.window_stride,
            "max_windows_per_scene": args.max_windows_per_scene,
            "n_example_points": args.n_example_points,
            "forgetting_factor": args.forgetting_factor,
            "initial_covariance": args.initial_covariance,
            "measurement_noise": args.measurement_noise,
            "include_fe_variants": not args.skip_fe_variants,
            "kalman_process_noise": args.kalman_process_noise,
            "fe_sgd_learning_rate": args.fe_sgd_learning_rate,
            "fe_sgd_momentum": args.fe_sgd_momentum,
            "fe_sgd_weight_decay": args.fe_sgd_weight_decay,
            "fe_window_size": args.fe_window_size,
            "fe_window_ridge": args.fe_window_ridge,
            "linear_include_bias": not args.linear_no_bias,
            "windows": [window.__dict__ for window in windows],
        },
    )

    from terrain_adaptation_rls.evaluation.baseline_sweep import run_baseline_sweep

    result = run_baseline_sweep(
        fe_run_dir=fe_run_dir,
        neuralfly_run_dir=neuralfly_run_dir,
        node_run_dir=node_run_dir,
        artifact_dir=run_dir,
        scenes=args.scenes,
        split=args.split,
        device=args.device,
        max_points=args.max_points,
        window_stride=args.window_stride,
        max_windows_per_scene=args.max_windows_per_scene,
        n_example_points=args.n_example_points,
        forgetting_factor=args.forgetting_factor,
        initial_covariance=args.initial_covariance,
        measurement_noise=args.measurement_noise,
        include_fe_variants=not args.skip_fe_variants,
        kalman_process_noise=args.kalman_process_noise,
        fe_sgd_learning_rate=args.fe_sgd_learning_rate,
        fe_sgd_momentum=args.fe_sgd_momentum,
        fe_sgd_weight_decay=args.fe_sgd_weight_decay,
        fe_window_size=args.fe_window_size,
        fe_window_ridge=args.fe_window_ridge,
        linear_include_bias=not args.linear_no_bias,
        progress=True,
    )
    print(result.artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
