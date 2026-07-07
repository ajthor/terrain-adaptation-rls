"""Run Van der Pol zero-start online adaptation comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="vanderpol_toy")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-example-points", type=int, default=64)
    parser.add_argument("--n-query-points", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--forgetting-factor", type=float, default=0.98)
    parser.add_argument("--initial-covariance", type=float, default=100.0)
    parser.add_argument("--measurement-noise", type=float, default=1e-5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the planned comparison without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            "valid Van der Pol toy evaluation: "
            "FE-ODE basis, FE-MLP basis, NeuralFly-style basis"
        )
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        run_dir / "command.json",
        {
            "command": "eval_vanderpol_toy",
            "device": args.device,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "n_basis": args.n_basis,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "n_example_points": args.n_example_points,
            "n_query_points": args.n_query_points,
            "ridge": args.ridge,
            "learning_rate": args.learning_rate,
            "dt": args.dt,
            "eval_steps": args.eval_steps,
            "forgetting_factor": args.forgetting_factor,
            "initial_covariance": args.initial_covariance,
            "measurement_noise": args.measurement_noise,
        },
    )

    from terrain_adaptation_rls.evaluation.vanderpol_toy import (
        run_vanderpol_toy_evaluation,
    )

    result = run_vanderpol_toy_evaluation(
        artifact_dir=run_dir,
        device=args.device,
        train_steps=args.train_steps,
        seed=args.seed,
        n_basis=args.n_basis,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        n_example_points=args.n_example_points,
        n_query_points=args.n_query_points,
        ridge=args.ridge,
        learning_rate=args.learning_rate,
        dt=args.dt,
        eval_steps=args.eval_steps,
        forgetting_factor=args.forgetting_factor,
        initial_covariance=args.initial_covariance,
        measurement_noise=args.measurement_noise,
    )
    print(result.artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
