"""Run Van der Pol zero-start online adaptation comparisons."""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from terrain_adaptation_rls.evaluation.artifacts import create_run_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="vanderpol_toy")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional devices for parallel seed sweeps, e.g. --devices cuda:0 cuda:1.",
    )
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds to run as one sweep. Overrides --seed.",
    )
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-example-points", type=int, default=64)
    parser.add_argument("--n-query-points", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--final-lr-fraction", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-trajectories-per-mu", type=int, default=4)
    parser.add_argument("--train-trajectories-per-mu", type=int, default=8)
    parser.add_argument("--train-trajectory-steps", type=int, default=160)
    parser.add_argument(
        "--no-restore-best",
        action="store_false",
        dest="restore_best",
        help="Keep the final training weights instead of restoring the best validation checkpoint.",
    )
    parser.set_defaults(restore_best=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--forgetting-factor", type=float, default=0.98)
    parser.add_argument("--initial-covariance", type=float, default=100.0)
    parser.add_argument("--measurement-noise", type=float, default=1e-5)
    parser.add_argument("--kalman-process-noise", type=float, default=0.0)
    parser.add_argument("--coefficient-sgd-learning-rate", type=float, default=1.0)
    parser.add_argument("--coefficient-sgd-momentum", type=float, default=0.0)
    parser.add_argument("--coefficient-sgd-weight-decay", type=float, default=0.0)
    parser.add_argument("--coefficient-window-size", type=int, default=64)
    parser.add_argument("--coefficient-window-ridge", type=float, default=1e-6)
    parser.add_argument("--maml-inner-learning-rate", type=float, default=1e-2)
    parser.add_argument("--maml-inner-steps", type=int, default=1)
    parser.add_argument(
        "--weak-fe-ode",
        action="store_true",
        help="Train the FE-ODE toy basis with an additional scheduled weak-form penalty.",
    )
    parser.add_argument("--weak-weight", type=float, default=0.01)
    parser.add_argument("--weak-start-step", type=int, default=1000)
    parser.add_argument("--weak-ramp-steps", type=int, default=500)
    parser.add_argument("--weak-window-points", type=int, default=128)
    parser.add_argument("--weak-test-functions", type=int, default=16)
    parser.add_argument("--weak-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--diagnostics-mode",
        choices=("all", "first", "none"),
        default="all",
        help="Control expensive per-seed plot artifacts when running a seed sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the planned comparison without training.",
    )
    return parser


def _run_seed_worker(job: dict[str, object]):
    from terrain_adaptation_rls.evaluation.vanderpol_toy import run_vanderpol_toy_evaluation

    return run_vanderpol_toy_evaluation(**job)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    devices = args.devices if args.devices is not None else [args.device]
    if args.dry_run:
        print(
            "valid Van der Pol toy evaluation: "
            "FE coefficient variants, NeuralFly, NODE-static, MAML-static/online; "
            f"seeds={seeds}, devices={devices}, "
            f"optimizer={args.optimizer}, lr_schedule={args.lr_schedule}"
        )
        return 0

    run_dir = create_run_dir(Path(args.output_root) / "eval", run_name=args.run_name)
    write_json(
        run_dir / "command.json",
        {
            "command": "eval_vanderpol_toy",
            "device": args.device,
            "devices": devices,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "seeds": seeds,
            "n_basis": args.n_basis,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "n_example_points": args.n_example_points,
            "n_query_points": args.n_query_points,
            "ridge": args.ridge,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "lr_schedule": args.lr_schedule,
            "warmup_steps": args.warmup_steps,
            "final_lr_fraction": args.final_lr_fraction,
            "validation_interval": args.validation_interval,
            "validation_batches": args.validation_batches,
            "validation_trajectories_per_mu": args.validation_trajectories_per_mu,
            "train_trajectories_per_mu": args.train_trajectories_per_mu,
            "train_trajectory_steps": args.train_trajectory_steps,
            "restore_best": args.restore_best,
            "dt": args.dt,
            "eval_steps": args.eval_steps,
            "forgetting_factor": args.forgetting_factor,
            "initial_covariance": args.initial_covariance,
            "measurement_noise": args.measurement_noise,
            "kalman_process_noise": args.kalman_process_noise,
            "coefficient_sgd_learning_rate": args.coefficient_sgd_learning_rate,
            "coefficient_sgd_momentum": args.coefficient_sgd_momentum,
            "coefficient_sgd_weight_decay": args.coefficient_sgd_weight_decay,
            "coefficient_window_size": args.coefficient_window_size,
            "coefficient_window_ridge": args.coefficient_window_ridge,
            "maml_inner_learning_rate": args.maml_inner_learning_rate,
            "maml_inner_steps": args.maml_inner_steps,
            "weak_fe_ode": args.weak_fe_ode,
            "weak_weight": args.weak_weight,
            "weak_start_step": args.weak_start_step,
            "weak_ramp_steps": args.weak_ramp_steps,
            "weak_window_points": args.weak_window_points,
            "weak_test_functions": args.weak_test_functions,
            "weak_ridge": args.weak_ridge,
            "diagnostics_mode": args.diagnostics_mode,
        },
    )

    from terrain_adaptation_rls.evaluation.vanderpol_toy import write_seed_sweep_summary

    jobs: list[dict[str, object]] = []
    for index, seed in enumerate(seeds):
        seed_dir = run_dir if len(seeds) == 1 else run_dir / f"seed_{seed}"
        write_diagnostics = args.diagnostics_mode == "all" or (
            args.diagnostics_mode == "first" and index == 0
        )
        jobs.append(
            {
                "artifact_dir": seed_dir,
                "device": devices[index % len(devices)],
                "train_steps": args.train_steps,
                "seed": seed,
                "n_basis": args.n_basis,
                "hidden_size": args.hidden_size,
                "batch_size": args.batch_size,
                "n_example_points": args.n_example_points,
                "n_query_points": args.n_query_points,
                "ridge": args.ridge,
                "learning_rate": args.learning_rate,
                "optimizer_name": args.optimizer,
                "weight_decay": args.weight_decay,
                "gradient_clip": args.gradient_clip,
                "lr_schedule": args.lr_schedule,
                "warmup_steps": args.warmup_steps,
                "final_lr_fraction": args.final_lr_fraction,
                "validation_interval": args.validation_interval,
                "validation_batches": args.validation_batches,
                "validation_trajectories_per_mu": args.validation_trajectories_per_mu,
                "train_trajectories_per_mu": args.train_trajectories_per_mu,
                "train_trajectory_steps": args.train_trajectory_steps,
                "restore_best": args.restore_best,
                "dt": args.dt,
                "eval_steps": args.eval_steps,
                "forgetting_factor": args.forgetting_factor,
                "initial_covariance": args.initial_covariance,
                "measurement_noise": args.measurement_noise,
                "kalman_process_noise": args.kalman_process_noise,
                "coefficient_sgd_learning_rate": args.coefficient_sgd_learning_rate,
                "coefficient_sgd_momentum": args.coefficient_sgd_momentum,
                "coefficient_sgd_weight_decay": args.coefficient_sgd_weight_decay,
                "coefficient_window_size": args.coefficient_window_size,
                "coefficient_window_ridge": args.coefficient_window_ridge,
                "maml_inner_learning_rate": args.maml_inner_learning_rate,
                "maml_inner_steps": args.maml_inner_steps,
                "weak_fe_ode": args.weak_fe_ode,
                "weak_weight": args.weak_weight,
                "weak_start_step": args.weak_start_step,
                "weak_ramp_steps": args.weak_ramp_steps,
                "weak_window_points": args.weak_window_points,
                "weak_test_functions": args.weak_test_functions,
                "weak_ridge": args.weak_ridge,
                "write_diagnostics": write_diagnostics,
            }
        )

    if len(jobs) > 1 and len(devices) > 1:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(jobs), len(devices)),
            mp_context=mp_context,
        ) as executor:
            futures = [executor.submit(_run_seed_worker, job) for job in jobs]
            results = [future.result() for future in as_completed(futures)]
    else:
        results = [_run_seed_worker(job) for job in jobs]
    results = sorted(results, key=lambda result: int(result.summary["seed"]))

    if len(results) > 1:
        write_seed_sweep_summary(run_dir, results)
        print(run_dir)
    else:
        print(results[0].artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
