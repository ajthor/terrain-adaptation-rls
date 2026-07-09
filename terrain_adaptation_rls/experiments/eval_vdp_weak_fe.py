"""Run the standalone weak-form FE Van der Pol sanity experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="vdp_weak_fe")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--n-basis", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--norm-weight", type=float, default=1e-3)
    parser.add_argument("--coeff-weight", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=41)
    parser.add_argument("--powers", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--example-starts", type=int, nargs="+", default=[0, 15, 30, 45])
    parser.add_argument("--query-starts", type=int, nargs="+", default=[60, 75, 90, 105])
    parser.add_argument(
        "--eval-example-starts",
        type=int,
        nargs="+",
        default=[0, 15, 30, 45, 60, 75],
    )
    parser.add_argument("--eval-mus", type=float, nargs="+", default=[0.6, 1.0, 1.7, 2.4])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = {
        "command": "eval_vdp_weak_fe",
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "noise": args.noise,
        "dt": args.dt,
        "steps": args.steps,
        "rollout_steps": args.rollout_steps,
        "n_basis": args.n_basis,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "ridge": args.ridge,
        "norm_weight": args.norm_weight,
        "coeff_weight": args.coeff_weight,
        "gradient_clip": args.gradient_clip,
        "window": args.window,
        "powers": args.powers,
        "example_starts": args.example_starts,
        "query_starts": args.query_starts,
        "eval_example_starts": args.eval_example_starts,
        "eval_mus": args.eval_mus,
        "write_plots": not args.no_plots,
    }
    if args.dry_run:
        print(
            "valid VDP weak FE experiment: "
            f"epochs={args.epochs}, n_basis={args.n_basis}, "
            f"window={args.window}, powers={args.powers}"
        )
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(run_dir / "command.json", payload)

    from terrain_adaptation_rls.evaluation.vdp_weak_fe import run_vdp_weak_fe_experiment

    run_vdp_weak_fe_experiment(
        artifact_dir=run_dir,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
        noise=args.noise,
        dt=args.dt,
        steps=args.steps,
        rollout_steps=args.rollout_steps,
        n_basis=args.n_basis,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        norm_weight=args.norm_weight,
        coeff_weight=args.coeff_weight,
        gradient_clip=args.gradient_clip,
        window=args.window,
        powers=tuple(args.powers),
        example_starts=tuple(args.example_starts),
        query_starts=tuple(args.query_starts),
        eval_mus=tuple(args.eval_mus),
        eval_example_starts=tuple(args.eval_example_starts),
        write_plots=not args.no_plots,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
