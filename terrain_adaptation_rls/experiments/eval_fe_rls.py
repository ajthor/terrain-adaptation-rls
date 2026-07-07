"""Direct FE-RLS streaming diagnostic command."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.configuration import load_config
from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-run-dir",
        required=True,
        help="Training artifact directory containing resolved_config.json and function_encoder_model.pth.",
    )
    parser.add_argument("--scene", default="scene1", help="Scene to stream through FE-RLS.")
    parser.add_argument("--run-name", default="fe_rls_streaming", help="Optional run name.")
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
        help="Number of same-scene example points for the offline FE comparison.",
    )
    parser.add_argument("--forgetting-factor", type=float, default=0.95)
    parser.add_argument("--initial-covariance", type=float, default=1000.0)
    parser.add_argument("--measurement-noise", type=float, default=1e-6)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the training run and arguments without creating artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_run_dir = Path(args.train_run_dir)
    _validate_train_run_dir(train_run_dir)
    config = load_config(train_run_dir / "resolved_config.json")

    if args.dry_run:
        print(
            f"valid FE-RLS eval: train_run={train_run_dir} "
            f"platform={config.platform} scene={args.scene}"
        )
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        run_dir / "command.json",
        {
            "command": "eval_fe_rls",
            "train_run_dir": train_run_dir,
            "scene": args.scene,
            "device": args.device,
            "max_points": args.max_points,
            "start_index": args.start_index,
            "n_example_points": args.n_example_points,
            "forgetting_factor": args.forgetting_factor,
            "initial_covariance": args.initial_covariance,
            "measurement_noise": args.measurement_noise,
        },
    )

    from terrain_adaptation_rls.evaluation.fe_rls import run_fe_rls_streaming_diagnostic

    run_fe_rls_streaming_diagnostic(
        train_run_dir=train_run_dir,
        artifact_dir=run_dir,
        scene=args.scene,
        device=args.device,
        max_points=args.max_points,
        start_index=args.start_index,
        n_example_points=args.n_example_points,
        forgetting_factor=args.forgetting_factor,
        initial_covariance=args.initial_covariance,
        measurement_noise=args.measurement_noise,
    )
    print(run_dir)
    return 0


def _validate_train_run_dir(path: Path) -> None:
    if not (path / "resolved_config.json").is_file():
        raise ValueError(f"{path} does not contain resolved_config.json")
    if not (path / "function_encoder_model.pth").is_file():
        raise ValueError(f"{path} does not contain function_encoder_model.pth")


if __name__ == "__main__":
    raise SystemExit(main())
