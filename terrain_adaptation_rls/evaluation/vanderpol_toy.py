"""Van der Pol toy-system comparisons for zero-start online adaptation."""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from terrain_adaptation_rls.estimators.linear import linear_predict
from terrain_adaptation_rls.estimators.linear import solve_ridge_coefficients
from terrain_adaptation_rls.methods.runtime import (
    ALPaCABasisProvider,
    Observation,
    RuntimeInput,
    TorchCoefficientMethod,
)
from terrain_adaptation_rls.evaluation.metrics import summarize_adaptation_time_metrics
from terrain_adaptation_rls.training.alpaca import (
    alpaca_loss,
    alpaca_prior_posterior,
    alpaca_update_posterior,
    predict_alpaca_features,
)
from terrain_adaptation_rls.training.weak_form import solve_weak_coefficients
from terrain_adaptation_rls.training.weak_form import weak_system_from_basis


@dataclass(frozen=True)
class VanDerPolTaskData:
    mu: float
    xs: torch.Tensor
    dt: torch.Tensor
    deltas: torch.Tensor


@dataclass(frozen=True)
class ToyEvaluationResult:
    artifact_dir: Path
    summary: dict[str, object]


class ToyMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FEIncrementBasis(torch.nn.Module):
    """FE-style basis with independent direct delta MLP basis functions."""

    def __init__(self, *, n_basis: int, hidden_size: int) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.basis_functions = torch.nn.ModuleList(
            [ToyMLP(3, 2, hidden_size) for _ in range(n_basis)]
        )

    @property
    def n_coeff(self) -> int:
        return self.n_basis

    def forward(self, inputs: RuntimeInput) -> torch.Tensor:
        z = _concat_state_dt(inputs)
        values = [basis(z) for basis in self.basis_functions]
        return torch.stack(values, dim=-1)


class FEODEBasis(torch.nn.Module):
    """FE-style basis with independent neural ODE vector-field basis functions."""

    def __init__(self, *, n_basis: int, hidden_size: int) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.vector_fields = torch.nn.ModuleList(
            [ToyMLP(2, 2, hidden_size) for _ in range(n_basis)]
        )

    @property
    def n_coeff(self) -> int:
        return self.n_basis

    def forward(self, inputs: RuntimeInput) -> torch.Tensor:
        xs = _as_batched_state(inputs.xs)
        dt = _as_batched_dt(inputs.dt, xs)
        values = [_rk4_delta(field, xs, dt) for field in self.vector_fields]
        return torch.stack(values, dim=-1)


class NeuralFlyToyBasis(torch.nn.Module):
    """NeuralFly-style shared feature map over ``(x, dt)``."""

    def __init__(self, *, n_basis: int, hidden_size: int) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * n_basis),
        )

    @property
    def n_coeff(self) -> int:
        return self.n_basis

    def forward(self, inputs: RuntimeInput) -> torch.Tensor:
        z = _concat_state_dt(inputs)
        features = self.network(z)
        return features.reshape(*z.shape[:-1], 2, self.n_basis)


class DirectNODEToyModel(torch.nn.Module):
    """Static neural ODE dynamics model mapping ``(x, dt)`` to ``delta x``."""

    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.vector_field = ToyMLP(2, 2, hidden_size)

    def forward(self, inputs: RuntimeInput) -> torch.Tensor:
        xs = _as_batched_state(inputs.xs)
        dt = _as_batched_dt(inputs.dt, xs)
        return _rk4_delta(self.vector_field, xs, dt)


def run_vanderpol_toy_evaluation(
    *,
    artifact_dir: str | Path,
    device: torch.device | str = "cpu",
    train_steps: int = 500,
    seed: int = 0,
    n_basis: int = 8,
    hidden_size: int = 64,
    batch_size: int = 8,
    n_example_points: int = 64,
    n_query_points: int = 64,
    ridge: float = 1e-5,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    lr_schedule: str = "cosine",
    warmup_steps: int = 100,
    final_lr_fraction: float = 0.1,
    validation_interval: int = 250,
    validation_batches: int = 8,
    validation_trajectories_per_mu: int = 4,
    restore_best: bool = True,
    dt: float = 0.02,
    train_mus: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
    test_mus: tuple[float, ...] = (1.25, 3.0),
    train_trajectories_per_mu: int = 8,
    train_trajectory_steps: int = 160,
    eval_steps: int = 500,
    forgetting_factor: float = 0.98,
    initial_covariance: float = 100.0,
    measurement_noise: float = 1e-5,
    kalman_process_noise: float = 0.0,
    coefficient_sgd_learning_rate: float = 1.0,
    coefficient_sgd_momentum: float = 0.0,
    coefficient_sgd_weight_decay: float = 0.0,
    coefficient_window_size: int = 64,
    coefficient_window_ridge: float = 1e-6,
    maml_inner_learning_rate: float = 1e-2,
    maml_inner_steps: int = 1,
    weak_fe_ode: bool = False,
    weak_weight: float = 0.01,
    weak_start_step: int = 1000,
    weak_ramp_steps: int = 500,
    weak_window_points: int = 128,
    weak_test_functions: int = 16,
    weak_ridge: float = 1e-4,
    write_diagnostics: bool = True,
) -> ToyEvaluationResult:
    """Train toy bases and evaluate zero-coefficient RLS on held-out VDP tasks."""

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(seed)

    train_data = generate_vanderpol_task_data(
        train_mus,
        trajectories_per_mu=train_trajectories_per_mu,
        steps=train_trajectory_steps,
        dt=dt,
        seed=seed,
        device=device,
    )
    validation_data = generate_vanderpol_task_data(
        train_mus,
        trajectories_per_mu=validation_trajectories_per_mu,
        steps=train_trajectory_steps,
        dt=dt,
        seed=seed + 100_000,
        device=device,
    )
    basis_models: dict[str, torch.nn.Module] = {
        "fe_ode": FEODEBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
        "fe_mlp": FEIncrementBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
        "neuralfly": NeuralFlyToyBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
        "alpaca": ALPaCABasisProvider(
            input_dim=3,
            output_dim=2,
            n_basis=n_basis,
            hidden_size=hidden_size,
            n_hidden_layers=2,
        ).to(device),
    }
    direct_models: dict[str, torch.nn.Module] = {
        "node_static": DirectNODEToyModel(hidden_size=hidden_size).to(device),
        "maml": DirectNODEToyModel(hidden_size=hidden_size).to(device),
    }
    labels = {
        "fe_ode": "FE-ODE basis",
        "fe_ode_static": "FE-ODE prior-static",
        "fe_ode_prior_rls": "FE-ODE prior-RLS",
        "fe_ode_rls": "FE-ODE RLS",
        "fe_ode_kalman": "FE-ODE Kalman",
        "fe_ode_sgd": "FE-ODE SGD",
        "fe_ode_window_ls": "FE-ODE window LS",
        "fe_mlp": "FE-MLP basis",
        "fe_mlp_rls": "FE-MLP RLS",
        "neuralfly": "NeuralFly basis",
        "neuralfly_static": "NeuralFly prior-static",
        "neuralfly_prior_rls": "NeuralFly prior-RLS",
        "neuralfly_rls": "NeuralFly RLS",
        "alpaca": "ALPaCA basis",
        "alpaca_rls": "ALPaCA online",
        "alpaca_static": "ALPaCA prior-static",
        "alpaca_online": "ALPaCA online",
        "node_static": "NODE static",
        "maml_static": "MAML static",
        "maml_online": "MAML online",
        "zero_delta": "zero state-change",
    }

    train_histories = {}
    for index, (name, model) in enumerate(basis_models.items()):
        if name == "alpaca":
            train_histories[name] = train_alpaca_toy_basis(
                model,
                train_data=train_data,
                steps=train_steps,
                batch_size=batch_size,
                n_example_points=n_example_points,
                n_query_points=n_query_points,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                gradient_clip=gradient_clip,
                lr_schedule=lr_schedule,
                warmup_steps=warmup_steps,
                final_lr_fraction=final_lr_fraction,
                validation_data=validation_data,
                validation_interval=validation_interval,
                validation_batches=validation_batches,
                restore_best=restore_best,
                seed=seed + index,
                device=device,
            )
            continue
        train_histories[name] = train_toy_basis(
            model,
            train_data=train_data,
            steps=train_steps,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            ridge=ridge,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            lr_schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
            validation_data=validation_data,
            validation_interval=validation_interval,
            validation_batches=validation_batches,
            restore_best=restore_best,
            seed=seed + index,
            device=device,
            weak_enabled=bool(weak_fe_ode and isinstance(model, FEODEBasis)),
            weak_weight=weak_weight,
            weak_start_step=weak_start_step,
            weak_ramp_steps=weak_ramp_steps,
            weak_window_points=weak_window_points,
            weak_test_functions=weak_test_functions,
            weak_ridge=weak_ridge,
            trajectory_steps=train_trajectory_steps,
        )
    train_histories["node_static"] = train_direct_toy_model(
        direct_models["node_static"],
        train_data=train_data,
        steps=train_steps,
        batch_size=batch_size,
        n_query_points=n_query_points,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
        validation_data=validation_data,
        validation_interval=validation_interval,
        validation_batches=validation_batches,
        restore_best=restore_best,
        seed=seed + 10_000,
        device=device,
    )
    train_histories["maml_static"] = train_maml_toy_model(
        direct_models["maml"],
        train_data=train_data,
        steps=train_steps,
        batch_size=batch_size,
        n_example_points=n_example_points,
        n_query_points=n_query_points,
        inner_learning_rate=maml_inner_learning_rate,
        inner_steps=maml_inner_steps,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
        validation_data=validation_data,
        validation_interval=validation_interval,
        validation_batches=validation_batches,
        restore_best=restore_best,
        seed=seed + 20_000,
        device=device,
    )
    prior_coefficients = {
        name: solve_global_train_coefficients(model, train_data=train_data, ridge=ridge)
        for name, model in basis_models.items()
        if name != "alpaca"
    }

    if write_diagnostics:
        for name, model in basis_models.items():
            write_basis_streamplots(
                artifact_path / f"basis_streamplots_{name}_rls.png",
                model=model,
                title=labels[f"{name}_rls"],
                dt=dt,
            )

    rows: list[dict[str, object]] = []
    scenarios: dict[str, object] = {}
    for mu in test_mus:
        scenario = _scenario_name(mu, train_mus)
        trajectory = generate_vanderpol_trajectory(
            mu=mu,
            steps=eval_steps,
            dt=dt,
            x0=torch.tensor([2.0, 0.0], dtype=torch.float32, device=device),
        )
        scenario_predictions: dict[str, torch.Tensor] = {}
        scenario_errors: dict[str, torch.Tensor] = {}
        scenario_coefficients: dict[str, torch.Tensor] = {}
        scenario_initial_coefficients: dict[str, torch.Tensor] = {}
        scenario_rows: list[dict[str, object]] = []

        coefficient_specs = [
            ("fe_ode_static", "fe_ode", "static", prior_coefficients["fe_ode"]),
            ("fe_ode_prior_rls", "fe_ode", "rls", prior_coefficients["fe_ode"]),
            ("fe_ode_rls", "fe_ode", "rls", None),
            ("fe_ode_kalman", "fe_ode", "kalman", None),
            ("fe_ode_sgd", "fe_ode", "sgd", None),
            ("fe_ode_window_ls", "fe_ode", "window_ls", None),
            ("fe_mlp_rls", "fe_mlp", "rls", None),
            ("neuralfly_static", "neuralfly", "static", prior_coefficients["neuralfly"]),
            (
                "neuralfly_prior_rls",
                "neuralfly",
                "rls",
                prior_coefficients["neuralfly"],
            ),
            ("neuralfly_rls", "neuralfly", "rls", None),
            ("alpaca_static", "alpaca", "static", None),
            ("alpaca_online", "alpaca", "online", None),
        ]
        for name, model_name, update_rule, initial_coefficients in coefficient_specs:
            model = basis_models[model_name]
            if model_name == "alpaca":
                result = evaluate_alpaca_method(
                    model,
                    trajectory=trajectory,
                    update_rule=update_rule,
                    recursive_horizons=(1, 5, 10, 25),
                )
            else:
                result = evaluate_coefficient_method(
                    model,
                    trajectory=trajectory,
                    update_rule=update_rule,
                    initial_coefficients=initial_coefficients,
                    forgetting_factor=forgetting_factor,
                    initial_covariance=initial_covariance,
                    measurement_noise=measurement_noise,
                    kalman_process_noise=kalman_process_noise,
                    coefficient_sgd_learning_rate=coefficient_sgd_learning_rate,
                    coefficient_sgd_momentum=coefficient_sgd_momentum,
                    coefficient_sgd_weight_decay=coefficient_sgd_weight_decay,
                    coefficient_window_size=coefficient_window_size,
                    coefficient_window_ridge=coefficient_window_ridge,
                    recursive_horizons=(1, 5, 10, 25),
                )
            add_method_result(
                rows=rows,
                scenario_rows=scenario_rows,
                scenario_predictions=scenario_predictions,
                scenario_errors=scenario_errors,
                scenario=scenario,
                mu=mu,
                name=name,
                label=labels[name],
                result=result,
            )
            scenario_coefficients[name] = result["coefficient_history"]
            scenario_initial_coefficients[name] = result["initial_coefficients"]

        node_result = evaluate_direct_model(
            direct_models["node_static"],
            trajectory=trajectory,
            recursive_horizons=(1, 5, 10, 25),
        )
        add_method_result(
            rows=rows,
            scenario_rows=scenario_rows,
            scenario_predictions=scenario_predictions,
            scenario_errors=scenario_errors,
            scenario=scenario,
            mu=mu,
            name="node_static",
            label=labels["node_static"],
            result=node_result,
        )
        maml_static_result = evaluate_direct_model(
            direct_models["maml"],
            trajectory=trajectory,
            recursive_horizons=(1, 5, 10, 25),
        )
        add_method_result(
            rows=rows,
            scenario_rows=scenario_rows,
            scenario_predictions=scenario_predictions,
            scenario_errors=scenario_errors,
            scenario=scenario,
            mu=mu,
            name="maml_static",
            label=labels["maml_static"],
            result=maml_static_result,
        )
        maml_online_result = evaluate_maml_online(
            direct_models["maml"],
            trajectory=trajectory,
            inner_learning_rate=maml_inner_learning_rate,
            inner_steps=maml_inner_steps,
            recursive_horizons=(1, 5, 10, 25),
            device=device,
        )
        add_method_result(
            rows=rows,
            scenario_rows=scenario_rows,
            scenario_predictions=scenario_predictions,
            scenario_errors=scenario_errors,
            scenario=scenario,
            mu=mu,
            name="maml_online",
            label=labels["maml_online"],
            result=maml_online_result,
        )

        zero_predictions = torch.zeros_like(trajectory["deltas"])
        zero_errors = torch.linalg.norm(zero_predictions - trajectory["deltas"], dim=-1)
        zero_metrics = summarize_online_errors(
            errors=zero_errors,
            predictions=zero_predictions,
            target=trajectory["deltas"],
        )
        zero_metrics.update(
            summarize_recursive_errors(
                predictor=None,
                trajectory=trajectory,
                coefficient_history=None,
                horizons=(1, 5, 10, 25),
            )
        )
        zero_row = {
            "scenario": scenario,
            "mu": mu,
            "method": "zero_delta",
            "label": labels["zero_delta"],
            **zero_metrics,
        }
        rows.append(zero_row)
        scenario_rows.append(zero_row)
        scenario_predictions["zero_delta"] = zero_predictions
        scenario_errors["zero_delta"] = zero_errors

        if write_diagnostics:
            write_online_error_plot(
                artifact_path / f"{scenario}_online_error.png",
                scenario=scenario,
                errors=scenario_errors,
                labels=labels,
            )
            write_online_component_error_plot(
                artifact_path / f"{scenario}_component_errors.png",
                scenario=scenario,
                predictions=scenario_predictions,
                target=trajectory["deltas"],
                labels=labels,
            )
            write_recursive_horizon_plot(
                artifact_path / f"{scenario}_recursive_horizon_errors.png",
                scenario=scenario,
                rows=scenario_rows,
            )
            write_phase_plot(
                artifact_path / f"{scenario}_phase_trajectory.png",
                scenario=scenario,
                trajectory=trajectory,
                predictions=scenario_predictions,
                labels=labels,
            )
            write_rollout_snapshot_plot(
                artifact_path / f"{scenario}_rollout_snapshots.png",
                scenario=scenario,
                trajectory=trajectory,
                methods={
                    method_name: basis_models[model_name]
                    for method_name, model_name, _, _ in coefficient_specs
                },
                coefficient_histories=scenario_coefficients,
                initial_coefficients=scenario_initial_coefficients,
                labels=labels,
            )
            write_streamplot_comparison(
                artifact_path / f"{scenario}_streamplots.png",
                scenario=scenario,
                mu=mu,
                trajectory=trajectory,
                methods={
                    method_name: basis_models[model_name]
                    for method_name, model_name, _, _ in coefficient_specs
                },
                coefficient_histories=scenario_coefficients,
                labels=labels,
                dt=dt,
            )
        scenarios[scenario] = {
            "mu": mu,
            "rows": scenario_rows,
        }

    write_rows_csv(artifact_path / "method_summary.csv", rows)
    if write_diagnostics:
        write_training_loss_plot(
            artifact_path / "training_losses.png",
            train_histories=train_histories,
            labels=labels,
        )
        write_metric_bar_grid(artifact_path / "metric_summary_grid.png", rows)
    summary = {
        "train_steps": train_steps,
        "seed": seed,
        "n_basis": n_basis,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "n_example_points": n_example_points,
        "n_query_points": n_query_points,
        "ridge": ridge,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "final_lr_fraction": final_lr_fraction,
        "validation_interval": validation_interval,
        "validation_batches": validation_batches,
        "validation_trajectories_per_mu": validation_trajectories_per_mu,
        "restore_best": restore_best,
        "dt": dt,
        "train_mus": list(train_mus),
        "test_mus": list(test_mus),
        "train_trajectories_per_mu": train_trajectories_per_mu,
        "train_trajectory_steps": train_trajectory_steps,
        "forgetting_factor": forgetting_factor,
        "initial_covariance": initial_covariance,
        "measurement_noise": measurement_noise,
        "kalman_process_noise": kalman_process_noise,
        "coefficient_sgd_learning_rate": coefficient_sgd_learning_rate,
        "coefficient_sgd_momentum": coefficient_sgd_momentum,
        "coefficient_sgd_weight_decay": coefficient_sgd_weight_decay,
        "coefficient_window_size": coefficient_window_size,
        "coefficient_window_ridge": coefficient_window_ridge,
        "maml_inner_learning_rate": maml_inner_learning_rate,
        "maml_inner_steps": maml_inner_steps,
        "weak_fe_ode": weak_fe_ode,
        "weak_weight": weak_weight,
        "weak_start_step": weak_start_step,
        "weak_ramp_steps": weak_ramp_steps,
        "weak_window_points": weak_window_points,
        "weak_test_functions": weak_test_functions,
        "weak_ridge": weak_ridge,
        "write_diagnostics": write_diagnostics,
        "train_histories": train_histories,
        "scenarios": scenarios,
    }
    (artifact_path / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return ToyEvaluationResult(artifact_dir=artifact_path, summary=summary)


def train_toy_basis(
    model: torch.nn.Module,
    *,
    train_data: dict[float, VanDerPolTaskData],
    steps: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    ridge: float,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    gradient_clip: float,
    lr_schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
    validation_data: dict[float, VanDerPolTaskData],
    validation_interval: int,
    validation_batches: int,
    restore_best: bool,
    seed: int,
    device: torch.device,
    weak_enabled: bool = False,
    weak_weight: float = 0.01,
    weak_start_step: int = 1000,
    weak_ramp_steps: int = 500,
    weak_window_points: int = 128,
    weak_test_functions: int = 16,
    weak_ridge: float = 1e-4,
    trajectory_steps: int = 160,
) -> dict[str, object]:
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 50_000)
    losses: list[float] = []
    pointwise_losses: list[float] = []
    weak_losses: list[float] = []
    learning_rates: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, steps + 1):
        lr_scale = learning_rate_scale(
            step=step,
            total_steps=steps,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        current_lr = learning_rate * lr_scale
        set_optimizer_lr(optimizer, current_lr)
        batch = sample_task_batch(
            train_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_with_solved_coefficients(model, batch, ridge=ridge)
        pointwise_loss = torch.nn.functional.mse_loss(prediction, batch["query_y"])
        weak_loss = None
        current_weak_weight = 0.0
        if weak_enabled and step >= weak_start_step:
            current_weak_weight = scheduled_weak_weight(
                weak_weight,
                step=step,
                start_step=weak_start_step,
                ramp_steps=weak_ramp_steps,
            )
            weak_batch = sample_weak_task_batch(
                train_data,
                batch_size=batch_size,
                window_points=weak_window_points,
                trajectory_steps=trajectory_steps,
                generator=generator,
                device=device,
            )
            weak_loss = weak_toy_fe_ode_loss(
                model,
                weak_batch,
                n_tests=weak_test_functions,
                ridge=weak_ridge,
            )
            loss = pointwise_loss + current_weak_weight * weak_loss
        else:
            loss = pointwise_loss
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pointwise_losses.append(float(pointwise_loss.detach().cpu()))
        if weak_loss is not None:
            weak_losses.append(float(weak_loss.detach().cpu()))
        learning_rates.append(current_lr)

        if should_validate(step, steps=steps, validation_interval=validation_interval):
            validation_loss = evaluate_toy_basis_loss(
                model,
                validation_data=validation_data,
                batches=validation_batches,
                batch_size=batch_size,
                n_example_points=n_example_points,
                n_query_points=n_query_points,
                ridge=ridge,
                generator=validation_generator,
                device=device,
            )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = clone_state_dict(model)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return {
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "pointwise_losses": pointwise_losses,
        "weak_losses": weak_losses,
        "learning_rates": learning_rates,
        "validation_steps": validation_steps,
        "validation_losses": validation_losses,
        "best_validation_loss": None
        if best_step is None
        else best_validation_loss,
        "best_step": best_step,
        "restored_best": bool(restore_best and best_state is not None),
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "final_lr_fraction": final_lr_fraction,
        "weak_enabled": weak_enabled,
        "weak_weight": weak_weight,
        "weak_start_step": weak_start_step,
        "weak_ramp_steps": weak_ramp_steps,
        "weak_window_points": weak_window_points,
        "weak_test_functions": weak_test_functions,
        "weak_ridge": weak_ridge,
    }


def train_alpaca_toy_basis(
    model: torch.nn.Module,
    *,
    train_data: dict[float, VanDerPolTaskData],
    steps: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    gradient_clip: float,
    lr_schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
    validation_data: dict[float, VanDerPolTaskData],
    validation_interval: int,
    validation_batches: int,
    restore_best: bool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 50_000)
    losses: list[float] = []
    learning_rates: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, steps + 1):
        lr_scale = learning_rate_scale(
            step=step,
            total_steps=steps,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        current_lr = learning_rate * lr_scale
        set_optimizer_lr(optimizer, current_lr)
        batch = sample_task_batch(
            train_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = alpaca_loss(model, task_batch_to_supervised_tuple(batch))
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        learning_rates.append(current_lr)

        if should_validate(step, steps=steps, validation_interval=validation_interval):
            validation_loss = evaluate_alpaca_toy_loss(
                model,
                validation_data=validation_data,
                batches=validation_batches,
                batch_size=batch_size,
                n_example_points=n_example_points,
                n_query_points=n_query_points,
                generator=validation_generator,
                device=device,
            )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = clone_state_dict(model)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return _training_history(
        losses=losses,
        learning_rates=learning_rates,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        restore_best=restore_best,
        best_state=best_state,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
    )


@torch.no_grad()
def evaluate_alpaca_toy_loss(
    model: torch.nn.Module,
    *,
    validation_data: dict[float, VanDerPolTaskData],
    batches: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    losses: list[float] = []
    was_training = model.training
    model.eval()
    for _ in range(max(batches, 1)):
        batch = sample_task_batch(
            validation_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        losses.append(float(alpaca_loss(model, task_batch_to_supervised_tuple(batch)).cpu()))
    model.train(was_training)
    return sum(losses) / len(losses)


def task_batch_to_supervised_tuple(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["query_x"],
        batch["query_dt"],
        batch["query_y"],
        batch["example_x"],
        batch["example_dt"],
        batch["example_y"],
    )


def predict_with_solved_coefficients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    ridge: float,
) -> torch.Tensor:
    example_features = model(RuntimeInput(batch["example_x"], batch["example_dt"]))
    coefficients = solve_ridge_coefficients(
        example_features,
        batch["example_y"],
        ridge=ridge,
    )
    query_features = model(RuntimeInput(batch["query_x"], batch["query_dt"]))
    return linear_predict(query_features, coefficients)


def weak_toy_fe_ode_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    n_tests: int,
    ridge: float,
) -> torch.Tensor:
    """Weak residual loss for the toy FE-ODE basis."""

    if not isinstance(model, FEODEBasis):
        raise TypeError("weak_toy_fe_ode_loss requires an FEODEBasis model")
    context_basis = evaluate_toy_raw_vector_field_basis(model, batch["context_x"])
    query_basis = evaluate_toy_raw_vector_field_basis(model, batch["query_x"])
    context_weak_basis, context_weak_target = weak_system_from_basis(
        batch["context_x"],
        context_basis,
        batch["context_dt"],
        n_tests=n_tests,
    )
    coefficients = solve_weak_coefficients(
        context_weak_basis,
        context_weak_target,
        ridge=ridge,
    )
    query_weak_basis, query_weak_target = weak_system_from_basis(
        batch["query_x"],
        query_basis,
        batch["query_dt"],
        n_tests=n_tests,
    )
    weak_prediction = torch.einsum("bmdk,bk->bmd", query_weak_basis, coefficients)
    return torch.nn.functional.mse_loss(weak_prediction, query_weak_target)


def evaluate_toy_raw_vector_field_basis(
    model: FEODEBasis,
    xs: torch.Tensor,
) -> torch.Tensor:
    values = [field(xs) for field in model.vector_fields]
    return torch.stack(values, dim=-1)


def scheduled_weak_weight(
    weak_weight: float,
    *,
    step: int,
    start_step: int,
    ramp_steps: int,
) -> float:
    if ramp_steps <= 0:
        return weak_weight
    ramp_position = min(max(step - start_step + 1, 0), ramp_steps)
    return weak_weight * ramp_position / ramp_steps


@torch.no_grad()
def solve_global_train_coefficients(
    model: torch.nn.Module,
    *,
    train_data: dict[float, VanDerPolTaskData],
    ridge: float,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for task in train_data.values():
        task_features = model(RuntimeInput(task.xs.unsqueeze(0), task.dt.unsqueeze(0)))
        features.append(task_features.squeeze(0))
        targets.append(task.deltas)
    feature_tensor = torch.cat(features, dim=0).unsqueeze(0)
    target_tensor = torch.cat(targets, dim=0).unsqueeze(0)
    return solve_ridge_coefficients(feature_tensor, target_tensor, ridge=ridge)


def train_direct_toy_model(
    model: torch.nn.Module,
    *,
    train_data: dict[float, VanDerPolTaskData],
    steps: int,
    batch_size: int,
    n_query_points: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    gradient_clip: float,
    lr_schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
    validation_data: dict[float, VanDerPolTaskData],
    validation_interval: int,
    validation_batches: int,
    restore_best: bool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 50_000)
    losses: list[float] = []
    learning_rates: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, steps + 1):
        lr_scale = learning_rate_scale(
            step=step,
            total_steps=steps,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        current_lr = learning_rate * lr_scale
        set_optimizer_lr(optimizer, current_lr)
        batch = sample_task_batch(
            train_data,
            batch_size=batch_size,
            n_example_points=0,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(RuntimeInput(batch["query_x"], batch["query_dt"]))
        loss = torch.nn.functional.mse_loss(prediction, batch["query_y"])
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        learning_rates.append(current_lr)

        if should_validate(step, steps=steps, validation_interval=validation_interval):
            validation_loss = evaluate_direct_toy_loss(
                model,
                validation_data=validation_data,
                batches=validation_batches,
                batch_size=batch_size,
                n_query_points=n_query_points,
                generator=validation_generator,
                device=device,
            )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = clone_state_dict(model)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return _training_history(
        losses=losses,
        learning_rates=learning_rates,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        restore_best=restore_best,
        best_state=best_state,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
    )


def train_maml_toy_model(
    model: torch.nn.Module,
    *,
    train_data: dict[float, VanDerPolTaskData],
    steps: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    inner_learning_rate: float,
    inner_steps: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    gradient_clip: float,
    lr_schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
    validation_data: dict[float, VanDerPolTaskData],
    validation_interval: int,
    validation_batches: int,
    restore_best: bool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 50_000)
    losses: list[float] = []
    learning_rates: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for step in range(1, steps + 1):
        lr_scale = learning_rate_scale(
            step=step,
            total_steps=steps,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            final_lr_fraction=final_lr_fraction,
        )
        current_lr = learning_rate * lr_scale
        set_optimizer_lr(optimizer, current_lr)
        batch = sample_task_batch(
            train_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = maml_batch_loss(
            model,
            batch=batch,
            inner_learning_rate=inner_learning_rate,
            inner_steps=inner_steps,
        )
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        learning_rates.append(current_lr)

        if should_validate(step, steps=steps, validation_interval=validation_interval):
            validation_loss = evaluate_maml_toy_loss(
                model,
                validation_data=validation_data,
                batches=validation_batches,
                batch_size=batch_size,
                n_example_points=n_example_points,
                n_query_points=n_query_points,
                inner_learning_rate=inner_learning_rate,
                inner_steps=inner_steps,
                generator=validation_generator,
                device=device,
            )
            validation_steps.append(step)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = clone_state_dict(model)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return _training_history(
        losses=losses,
        learning_rates=learning_rates,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        restore_best=restore_best,
        best_state=best_state,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        final_lr_fraction=final_lr_fraction,
        inner_learning_rate=inner_learning_rate,
        inner_steps=inner_steps,
    )


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {optimizer_name}")


def learning_rate_scale(
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return max(step / warmup_steps, 1e-6)
    if schedule == "none":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    decay_steps = max(total_steps - max(warmup_steps, 0), 1)
    decay_step = min(max(step - max(warmup_steps, 0), 0), decay_steps)
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_fraction + (1.0 - final_lr_fraction) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def should_validate(step: int, *, steps: int, validation_interval: int) -> bool:
    if validation_interval <= 0:
        return step == steps
    return step == steps or step % validation_interval == 0


@torch.no_grad()
def evaluate_toy_basis_loss(
    model: torch.nn.Module,
    *,
    validation_data: dict[float, VanDerPolTaskData],
    batches: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    ridge: float,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    losses: list[float] = []
    for _ in range(max(batches, 1)):
        batch = sample_task_batch(
            validation_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        prediction = predict_with_solved_coefficients(model, batch, ridge=ridge)
        losses.append(float(torch.nn.functional.mse_loss(prediction, batch["query_y"]).cpu()))
    return _mean(losses)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


@torch.no_grad()
def evaluate_direct_toy_loss(
    model: torch.nn.Module,
    *,
    validation_data: dict[float, VanDerPolTaskData],
    batches: int,
    batch_size: int,
    n_query_points: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    losses: list[float] = []
    for _ in range(max(batches, 1)):
        batch = sample_task_batch(
            validation_data,
            batch_size=batch_size,
            n_example_points=0,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        prediction = model(RuntimeInput(batch["query_x"], batch["query_dt"]))
        losses.append(float(torch.nn.functional.mse_loss(prediction, batch["query_y"]).cpu()))
    return _mean(losses)


def maml_batch_loss(
    model: torch.nn.Module,
    *,
    batch: dict[str, torch.Tensor],
    inner_learning_rate: float,
    inner_steps: int,
) -> torch.Tensor:
    task_losses = []
    for task_index in range(batch["example_x"].shape[0]):
        adapted_parameters = _adapt_functional_parameters(
            model,
            xs=batch["example_x"][task_index : task_index + 1],
            dt=batch["example_dt"][task_index : task_index + 1],
            target=batch["example_y"][task_index : task_index + 1],
            inner_learning_rate=inner_learning_rate,
            inner_steps=inner_steps,
        )
        query_prediction = _functional_forward(
            model,
            adapted_parameters,
            RuntimeInput(
                batch["query_x"][task_index : task_index + 1],
                batch["query_dt"][task_index : task_index + 1],
            ),
        )
        task_losses.append(
            torch.nn.functional.mse_loss(
                query_prediction,
                batch["query_y"][task_index : task_index + 1],
            )
        )
    return torch.stack(task_losses).mean()


def evaluate_maml_toy_loss(
    model: torch.nn.Module,
    *,
    validation_data: dict[float, VanDerPolTaskData],
    batches: int,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    inner_learning_rate: float,
    inner_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    losses: list[float] = []
    was_training = model.training
    model.eval()
    for _ in range(max(batches, 1)):
        batch = sample_task_batch(
            validation_data,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            generator=generator,
            device=device,
        )
        with torch.enable_grad():
            loss = maml_batch_loss(
                model,
                batch=batch,
                inner_learning_rate=inner_learning_rate,
                inner_steps=inner_steps,
            )
        losses.append(float(loss.detach().cpu()))
    model.train(was_training)
    return _mean(losses)


def _training_history(
    *,
    losses: list[float],
    learning_rates: list[float],
    validation_steps: list[int],
    validation_losses: list[float],
    best_validation_loss: float,
    best_step: int | None,
    restore_best: bool,
    best_state: dict[str, torch.Tensor] | None,
    optimizer_name: str,
    weight_decay: float,
    gradient_clip: float,
    lr_schedule: str,
    warmup_steps: int,
    final_lr_fraction: float,
    **extra: object,
) -> dict[str, object]:
    return {
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "learning_rates": learning_rates,
        "validation_steps": validation_steps,
        "validation_losses": validation_losses,
        "best_validation_loss": None
        if best_step is None
        else best_validation_loss,
        "best_step": best_step,
        "restored_best": bool(restore_best and best_state is not None),
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "final_lr_fraction": final_lr_fraction,
        **extra,
    }


def _functional_forward(
    model: torch.nn.Module,
    parameters: dict[str, torch.Tensor],
    inputs: RuntimeInput,
) -> torch.Tensor:
    try:
        from torch.func import functional_call
    except ImportError:  # pragma: no cover - for older torch builds.
        from torch.nn.utils.stateless import functional_call

    return functional_call(model, parameters, (inputs,))


def _adapt_functional_parameters(
    model: torch.nn.Module,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    inner_learning_rate: float,
    inner_steps: int,
) -> dict[str, torch.Tensor]:
    adapted = {name: value for name, value in model.named_parameters()}
    for _ in range(max(inner_steps, 0)):
        prediction = _functional_forward(model, adapted, RuntimeInput(xs, dt))
        loss = torch.nn.functional.mse_loss(prediction, target)
        gradients = torch.autograd.grad(
            loss,
            tuple(adapted.values()),
            create_graph=False,
            allow_unused=False,
        )
        adapted = {
            name: parameter - inner_learning_rate * gradient.detach()
            for (name, parameter), gradient in zip(adapted.items(), gradients)
        }
    return adapted


def add_method_result(
    *,
    rows: list[dict[str, object]],
    scenario_rows: list[dict[str, object]],
    scenario_predictions: dict[str, torch.Tensor],
    scenario_errors: dict[str, torch.Tensor],
    scenario: str,
    mu: float,
    name: str,
    label: str,
    result: dict[str, object],
) -> None:
    row = {
        "scenario": scenario,
        "mu": mu,
        "method": name,
        "label": label,
        **result["metrics"],
    }
    rows.append(row)
    scenario_rows.append(row)
    scenario_predictions[name] = result["predictions"]
    scenario_errors[name] = result["errors"]


def _initial_coefficients_tensor(
    *,
    n_coeff: int,
    reference: torch.Tensor,
    initial_coefficients: torch.Tensor | None,
) -> torch.Tensor:
    if initial_coefficients is None:
        return torch.zeros(1, n_coeff, dtype=reference.dtype, device=reference.device)
    coefficients = initial_coefficients.to(dtype=reference.dtype, device=reference.device)
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)
    if coefficients.shape != (1, n_coeff):
        raise ValueError(
            "initial_coefficients must have shape [n_coeff] or [1, n_coeff], "
            f"got {tuple(coefficients.shape)} for n_coeff={n_coeff}"
        )
    return coefficients


@torch.no_grad()
def evaluate_coefficient_method(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    update_rule: str,
    initial_coefficients: torch.Tensor | None,
    forgetting_factor: float,
    initial_covariance: float,
    measurement_noise: float,
    kalman_process_noise: float,
    coefficient_sgd_learning_rate: float,
    coefficient_sgd_momentum: float,
    coefficient_sgd_weight_decay: float,
    coefficient_window_size: int,
    coefficient_window_ridge: float,
    recursive_horizons: tuple[int, ...],
) -> dict[str, object]:
    xs = trajectory["xs"]
    dt = trajectory["dt"]
    target = trajectory["deltas"]
    n_coeff = int(model.n_basis)
    initial_coefficients = _initial_coefficients_tensor(
        n_coeff=n_coeff,
        reference=xs,
        initial_coefficients=initial_coefficients,
    )
    method = TorchCoefficientMethod(
        model,
        update_rule="rls" if update_rule == "static" else update_rule,
        output_dim=2,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        process_noise=kalman_process_noise,
        learning_rate=coefficient_sgd_learning_rate,
        momentum=coefficient_sgd_momentum,
        weight_decay=coefficient_sgd_weight_decay,
        window_size=coefficient_window_size,
        ridge=coefficient_window_ridge,
        initial_coefficients=initial_coefficients,
        device=xs.device,
        dtype=xs.dtype,
    )
    state = method.initial_state()
    predictions: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    for index in range(xs.shape[0]):
        inputs = RuntimeInput(
            xs[index : index + 1].unsqueeze(0),
            dt[index : index + 1].unsqueeze(0),
        )
        prediction = method.predict(state, inputs).squeeze(0).squeeze(0)
        if update_rule != "static":
            state = method.update(
                state,
                Observation(inputs=inputs, target=target[index : index + 1].unsqueeze(0)),
            )
        predictions.append(prediction.detach())
        coefficients.append(state.coefficients.squeeze(0).detach())

    prediction_tensor = torch.stack(predictions)
    errors = torch.linalg.norm(prediction_tensor - target, dim=-1)
    coefficient_history = torch.stack(coefficients)
    metrics = summarize_online_errors(
        errors=errors,
        predictions=prediction_tensor,
        target=target,
    )
    metrics.update(
        summarize_recursive_errors(
            predictor=model,
            trajectory=trajectory,
            coefficient_history=coefficient_history,
            initial_coefficients=initial_coefficients.squeeze(0),
            horizons=recursive_horizons,
        )
    )
    metrics.update(
        {
            "final_coefficient_norm": float(torch.linalg.norm(coefficient_history[-1])),
            "mean_coefficient_norm": float(torch.linalg.norm(coefficient_history, dim=-1).mean()),
        }
    )
    return {
        "predictions": prediction_tensor.cpu(),
        "errors": errors.cpu(),
        "coefficient_history": coefficient_history.cpu(),
        "initial_coefficients": initial_coefficients.squeeze(0).detach().cpu(),
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_alpaca_method(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    update_rule: str,
    recursive_horizons: tuple[int, ...],
) -> dict[str, object]:
    xs = trajectory["xs"]
    dt = trajectory["dt"]
    target = trajectory["deltas"]
    posterior = alpaca_prior_posterior(model, batch_size=1)
    initial_coefficients = posterior.mean.squeeze(0).detach()
    predictions: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []

    for index in range(xs.shape[0]):
        inputs = RuntimeInput(
            xs[index : index + 1].unsqueeze(0),
            dt[index : index + 1].unsqueeze(0),
        )
        features = model(inputs)
        prediction = predict_alpaca_features(features, posterior.mean).squeeze(0).squeeze(0)
        if update_rule != "static":
            posterior = alpaca_update_posterior(
                model,
                posterior,
                features,
                target[index : index + 1].unsqueeze(0),
            )
        predictions.append(prediction.detach())
        coefficients.append(posterior.mean.squeeze(0).detach())

    prediction_tensor = torch.stack(predictions)
    errors = torch.linalg.norm(prediction_tensor - target, dim=-1)
    coefficient_history = torch.stack(coefficients)
    metrics = summarize_online_errors(
        errors=errors,
        predictions=prediction_tensor,
        target=target,
    )
    metrics.update(
        summarize_recursive_errors(
            predictor=model,
            trajectory=trajectory,
            coefficient_history=coefficient_history,
            initial_coefficients=initial_coefficients,
            horizons=recursive_horizons,
        )
    )
    metrics.update(
        {
            "final_coefficient_norm": float(torch.linalg.norm(coefficient_history[-1])),
            "mean_coefficient_norm": float(torch.linalg.norm(coefficient_history, dim=-1).mean()),
        }
    )
    return {
        "predictions": prediction_tensor.cpu(),
        "errors": errors.cpu(),
        "coefficient_history": coefficient_history.cpu(),
        "initial_coefficients": initial_coefficients.cpu(),
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_direct_model(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    recursive_horizons: tuple[int, ...],
) -> dict[str, object]:
    xs = trajectory["xs"]
    dt = trajectory["dt"]
    target = trajectory["deltas"]
    was_training = model.training
    model.eval()
    predictions = model(RuntimeInput(xs.unsqueeze(0), dt.unsqueeze(0))).squeeze(0)
    model.train(was_training)
    errors = torch.linalg.norm(predictions - target, dim=-1)
    metrics = summarize_online_errors(errors=errors, predictions=predictions, target=target)
    metrics.update(
        summarize_recursive_direct_errors(
            model,
            trajectory=trajectory,
            horizons=recursive_horizons,
        )
    )
    metrics.update(
        {
            "final_coefficient_norm": float("nan"),
            "mean_coefficient_norm": float("nan"),
        }
    )
    return {
        "predictions": predictions.cpu(),
        "errors": errors.cpu(),
        "metrics": metrics,
    }


def evaluate_maml_online(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    inner_learning_rate: float,
    inner_steps: int,
    recursive_horizons: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    xs = trajectory["xs"]
    dt = trajectory["dt"]
    target = trajectory["deltas"]
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_learning_rate)
    predictions: list[torch.Tensor] = []
    state_history: list[dict[str, torch.Tensor]] = []

    for index in range(xs.shape[0]):
        inputs = RuntimeInput(
            xs[index : index + 1].unsqueeze(0),
            dt[index : index + 1].unsqueeze(0),
        )
        with torch.no_grad():
            prediction = adapted(inputs).squeeze(0).squeeze(0)
        predictions.append(prediction.detach())
        for _ in range(max(inner_steps, 0)):
            optimizer.zero_grad(set_to_none=True)
            update_prediction = adapted(inputs).squeeze(0)
            loss = torch.nn.functional.mse_loss(
                update_prediction,
                target[index : index + 1],
            )
            loss.backward()
            optimizer.step()
        state_history.append(clone_state_dict(adapted))

    prediction_tensor = torch.stack(predictions)
    errors = torch.linalg.norm(prediction_tensor - target, dim=-1)
    metrics = summarize_online_errors(
        errors=errors,
        predictions=prediction_tensor,
        target=target,
    )
    metrics.update(
        summarize_recursive_direct_errors(
            model,
            trajectory=trajectory,
            horizons=recursive_horizons,
            state_history=state_history,
        )
    )
    metrics.update(
        {
            "final_coefficient_norm": float("nan"),
            "mean_coefficient_norm": float("nan"),
        }
    )
    return {
        "predictions": prediction_tensor.cpu(),
        "errors": errors.cpu(),
        "metrics": metrics,
    }


def summarize_online_errors(
    *,
    errors: torch.Tensor,
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    zero_error = torch.linalg.norm(target, dim=-1)
    metrics = {
        "mean_error": float(errors.mean()),
        "first_10_mean_error": _prefix_mean(errors, 10),
        "first_25_mean_error": _prefix_mean(errors, 25),
        "first_50_mean_error": _prefix_mean(errors, 50),
        "last_50_mean_error": _suffix_mean(errors, 50),
        "final_accumulated_error": float(errors.sum()),
        "mse": float(torch.nn.functional.mse_loss(predictions, target)),
        "mean_error_to_zero_delta_ratio": _safe_ratio(float(errors.mean()), float(zero_error.mean())),
        "first_25_error_to_zero_delta_ratio": _safe_ratio(
            _prefix_mean(errors, 25),
            _prefix_mean(zero_error, 25),
        ),
    }
    metrics.update(summarize_adaptation_time_metrics(errors, window=10))
    return metrics


@torch.no_grad()
def summarize_recursive_errors(
    *,
    predictor: torch.nn.Module | None,
    trajectory: dict[str, torch.Tensor],
    coefficient_history: torch.Tensor | None,
    horizons: tuple[int, ...],
    initial_coefficients: torch.Tensor | None = None,
    max_rollouts: int = 64,
) -> dict[str, float]:
    states = trajectory["states"]
    dt = trajectory["dt"]
    valid_horizons = sorted({horizon for horizon in horizons if 0 < horizon < states.shape[0]})
    if not valid_horizons:
        return {}
    max_horizon = max(valid_horizons)
    max_start = states.shape[0] - max_horizon - 1
    starts = torch.linspace(
        0,
        max_start,
        steps=min(max_rollouts, max_start + 1),
        device=states.device,
    ).round().to(torch.long).unique(sorted=True)
    horizon_errors: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}
    horizon_accumulated: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}

    for start_tensor in starts:
        start = int(start_tensor.item())
        current = states[start].clone()
        step_errors: list[float] = []
        coefficients = None
        if coefficient_history is not None:
            if start == 0:
                if initial_coefficients is None:
                    coefficients = torch.zeros_like(coefficient_history[0]).unsqueeze(0)
                else:
                    coefficients = initial_coefficients.to(
                        device=states.device,
                        dtype=states.dtype,
                    ).unsqueeze(0)
            else:
                coefficients = coefficient_history[start - 1].unsqueeze(0)
        for offset in range(max_horizon):
            index = start + offset
            if predictor is None:
                delta = torch.zeros_like(current)
            else:
                features = predictor(
                    RuntimeInput(
                        current.reshape(1, 1, -1),
                        dt[index : index + 1].reshape(1, 1),
                    )
                )
                assert coefficients is not None
                delta = linear_predict(features, coefficients).squeeze(0).squeeze(0)
            current = current + delta
            error = float(torch.linalg.norm(current - states[index + 1]))
            step_errors.append(error)
            horizon = offset + 1
            if horizon in horizon_errors:
                horizon_errors[horizon].append(error)
                horizon_accumulated[horizon].append(sum(step_errors))

    metrics: dict[str, float] = {}
    for horizon in valid_horizons:
        prefix = f"recursive_k{horizon}"
        metrics[f"{prefix}_final_step_error_mean"] = _mean(horizon_errors[horizon])
        metrics[f"{prefix}_accumulated_error_mean"] = _mean(horizon_accumulated[horizon])
    return metrics


@torch.no_grad()
def summarize_recursive_direct_errors(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    horizons: tuple[int, ...],
    max_rollouts: int = 64,
    state_history: list[dict[str, torch.Tensor]] | None = None,
) -> dict[str, float]:
    states = trajectory["states"]
    dt = trajectory["dt"]
    valid_horizons = sorted({horizon for horizon in horizons if 0 < horizon < states.shape[0]})
    if not valid_horizons:
        return {}
    max_horizon = max(valid_horizons)
    max_start = states.shape[0] - max_horizon - 1
    starts = torch.linspace(
        0,
        max_start,
        steps=min(max_rollouts, max_start + 1),
        device=states.device,
    ).round().to(torch.long).unique(sorted=True)
    horizon_errors: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}
    horizon_accumulated: dict[int, list[float]] = {horizon: [] for horizon in valid_horizons}
    base_state = clone_state_dict(model)

    for start_tensor in starts:
        start = int(start_tensor.item())
        rollout_model = model
        if state_history is not None:
            rollout_model = copy.deepcopy(model).to(states.device)
            rollout_model.load_state_dict(base_state if start == 0 else state_history[start - 1])
        rollout_model.eval()
        current = states[start].clone()
        step_errors: list[float] = []
        for offset in range(max_horizon):
            index = start + offset
            delta = rollout_model(
                RuntimeInput(
                    current.reshape(1, 1, -1),
                    dt[index : index + 1].reshape(1, 1),
                )
            ).squeeze(0).squeeze(0)
            current = current + delta
            error = float(torch.linalg.norm(current - states[index + 1]))
            step_errors.append(error)
            horizon = offset + 1
            if horizon in horizon_errors:
                horizon_errors[horizon].append(error)
                horizon_accumulated[horizon].append(sum(step_errors))

    metrics: dict[str, float] = {}
    for horizon in valid_horizons:
        prefix = f"recursive_k{horizon}"
        metrics[f"{prefix}_final_step_error_mean"] = _mean(horizon_errors[horizon])
        metrics[f"{prefix}_accumulated_error_mean"] = _mean(horizon_accumulated[horizon])
    return metrics


def generate_vanderpol_task_data(
    mus: Iterable[float],
    *,
    trajectories_per_mu: int,
    steps: int,
    dt: float,
    seed: int,
    device: torch.device,
) -> dict[float, VanDerPolTaskData]:
    generator = torch.Generator(device=device).manual_seed(seed)
    data: dict[float, VanDerPolTaskData] = {}
    for mu in mus:
        xs_parts: list[torch.Tensor] = []
        dt_parts: list[torch.Tensor] = []
        delta_parts: list[torch.Tensor] = []
        for _ in range(trajectories_per_mu):
            x0 = 4.0 * torch.rand(2, generator=generator, device=device) - 2.0
            trajectory = generate_vanderpol_trajectory(mu=mu, steps=steps, dt=dt, x0=x0)
            xs_parts.append(trajectory["xs"])
            dt_parts.append(trajectory["dt"])
            delta_parts.append(trajectory["deltas"])
        data[float(mu)] = VanDerPolTaskData(
            mu=float(mu),
            xs=torch.cat(xs_parts, dim=0),
            dt=torch.cat(dt_parts, dim=0),
            deltas=torch.cat(delta_parts, dim=0),
        )
    return data


def generate_vanderpol_trajectory(
    *,
    mu: float,
    steps: int,
    dt: float,
    x0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    states = [x0]
    current = x0
    dt_tensor = torch.tensor(dt, dtype=x0.dtype, device=x0.device)
    for _ in range(steps):
        current = current + _true_vanderpol_delta(current, mu=mu, dt=dt_tensor)
        states.append(current)
    state_tensor = torch.stack(states)
    xs = state_tensor[:-1]
    next_states = state_tensor[1:]
    return {
        "states": state_tensor,
        "xs": xs,
        "dt": torch.full((steps,), dt, dtype=x0.dtype, device=x0.device),
        "deltas": next_states - xs,
    }


def sample_task_batch(
    train_data: dict[float, VanDerPolTaskData],
    *,
    batch_size: int,
    n_example_points: int,
    n_query_points: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tasks = list(train_data.values())
    example_xs: list[torch.Tensor] = []
    example_dt: list[torch.Tensor] = []
    example_y: list[torch.Tensor] = []
    query_xs: list[torch.Tensor] = []
    query_dt: list[torch.Tensor] = []
    query_y: list[torch.Tensor] = []
    for _ in range(batch_size):
        task_index = int(torch.randint(len(tasks), (1,), generator=generator, device=device))
        task = tasks[task_index]
        indices = torch.randperm(task.xs.shape[0], generator=generator, device=device)
        example_indices = indices[:n_example_points]
        query_indices = indices[n_example_points : n_example_points + n_query_points]
        example_xs.append(task.xs[example_indices])
        example_dt.append(task.dt[example_indices])
        example_y.append(task.deltas[example_indices])
        query_xs.append(task.xs[query_indices])
        query_dt.append(task.dt[query_indices])
        query_y.append(task.deltas[query_indices])
    return {
        "example_x": torch.stack(example_xs),
        "example_dt": torch.stack(example_dt),
        "example_y": torch.stack(example_y),
        "query_x": torch.stack(query_xs),
        "query_dt": torch.stack(query_dt),
        "query_y": torch.stack(query_y),
    }


def sample_weak_task_batch(
    train_data: dict[float, VanDerPolTaskData],
    *,
    batch_size: int,
    window_points: int,
    trajectory_steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample context/query trajectory windows without crossing rollout boundaries."""

    if window_points <= 1:
        raise ValueError("weak window must contain at least two points")
    if trajectory_steps < window_points:
        raise ValueError("trajectory_steps must be at least weak window_points")

    tasks = list(train_data.values())
    context_xs: list[torch.Tensor] = []
    context_dt: list[torch.Tensor] = []
    query_xs: list[torch.Tensor] = []
    query_dt: list[torch.Tensor] = []
    for _ in range(batch_size):
        task_index = int(torch.randint(len(tasks), (1,), generator=generator, device=device))
        task = tasks[task_index]
        context_slice = sample_task_window_slice(
            task,
            trajectory_steps=trajectory_steps,
            window_points=window_points,
            generator=generator,
            device=device,
        )
        query_slice = sample_task_window_slice(
            task,
            trajectory_steps=trajectory_steps,
            window_points=window_points,
            generator=generator,
            device=device,
        )
        context_xs.append(task.xs[context_slice])
        context_dt.append(task.dt[context_slice])
        query_xs.append(task.xs[query_slice])
        query_dt.append(task.dt[query_slice])

    return {
        "context_x": torch.stack(context_xs),
        "context_dt": torch.stack(context_dt),
        "query_x": torch.stack(query_xs),
        "query_dt": torch.stack(query_dt),
    }


def sample_task_window_slice(
    task: VanDerPolTaskData,
    *,
    trajectory_steps: int,
    window_points: int,
    generator: torch.Generator,
    device: torch.device,
) -> slice:
    n_trajectories = task.xs.shape[0] // trajectory_steps
    if n_trajectories <= 0:
        raise ValueError("task does not contain a full trajectory")
    trajectory_index = int(torch.randint(n_trajectories, (1,), generator=generator, device=device))
    offset = int(
        torch.randint(
            trajectory_steps - window_points + 1,
            (1,),
            generator=generator,
            device=device,
        )
    )
    start = trajectory_index * trajectory_steps + offset
    return slice(start, start + window_points)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_seed_sweep_summary(
    artifact_dir: str | Path,
    results: list[ToyEvaluationResult],
) -> None:
    artifact_path = Path(artifact_dir)
    seed_rows: list[dict[str, object]] = []
    for result in results:
        seed = result.summary["seed"]
        scenarios = result.summary["scenarios"]
        assert isinstance(scenarios, dict)
        for scenario_summary in scenarios.values():
            assert isinstance(scenario_summary, dict)
            rows = scenario_summary["rows"]
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, dict)
                seed_rows.append({"seed": seed, **row})

    aggregate_rows = summarize_seed_sweep_rows(seed_rows)
    write_rows_csv(artifact_path / "seed_method_summary.csv", seed_rows)
    write_rows_csv(artifact_path / "aggregate_method_summary.csv", aggregate_rows)
    write_seed_aggregate_metric_plot(
        artifact_path / "aggregate_metric_summary.png",
        aggregate_rows,
    )
    (artifact_path / "aggregate_summary.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "n_seed_rows": len(seed_rows),
                    "n_aggregate_rows": len(aggregate_rows),
                    "aggregate_rows": aggregate_rows,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def summarize_seed_sweep_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["scenario"]),
            float(row["mu"]),
            str(row["method"]),
            str(row["label"]),
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (scenario, mu, method, label), group_rows in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "scenario": scenario,
            "mu": mu,
            "method": method,
            "label": label,
            "n_seeds": len({int(row["seed"]) for row in group_rows}),
        }
        metric_names = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if key not in {"seed", "scenario", "mu", "method", "label"}
                and _is_float_like(value)
            }
        )
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows if metric in row]
            finite_values = [value for value in values if math.isfinite(value)]
            aggregate[f"{metric}_mean"] = _mean(finite_values) if finite_values else float("nan")
            aggregate[f"{metric}_std"] = _std(finite_values)
            aggregate[f"{metric}_nonfinite_count"] = len(values) - len(finite_values)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def write_seed_aggregate_metric_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = []
    for row in rows:
        scenario = str(row["scenario"])
        if scenario not in scenarios:
            scenarios.append(scenario)
    metrics = [
        ("first_25_mean_error_mean", "first 25 online steps"),
        ("adaptation_samples_to_25pct_improvement_mean", "samples to 25% adaptation"),
        ("mean_error_mean", "all online steps"),
        ("recursive_k10_accumulated_error_mean_mean", "recursive k=10 accumulated"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(scenarios),
        figsize=(6 * len(scenarios), 2.7 * len(metrics)),
    )
    if len(scenarios) == 1:
        axes = [[ax] for ax in axes]
    for row_index, (metric, title) in enumerate(metrics):
        for col_index, scenario in enumerate(scenarios):
            ax = axes[row_index][col_index]
            scenario_rows = [
                row
                for row in rows
                if row["scenario"] == scenario and _is_float_like(row.get(metric))
            ]
            scenario_rows = sorted(scenario_rows, key=lambda row: float(row[metric]))
            labels = [str(row["label"]) for row in scenario_rows]
            values = [float(row[metric]) for row in scenario_rows]
            std_key = metric.replace("_mean", "_std")
            errors = [float(row.get(std_key, 0.0)) for row in scenario_rows]
            ax.bar(range(len(labels)), values, yerr=errors, capsize=3)
            finite_values = [value for value in values if math.isfinite(value) and value > 0.0]
            if finite_values and max(finite_values) / max(min(finite_values), 1e-12) > 100.0:
                ax.set_yscale("log")
                ax.set_ylabel("error (log scale)")
            ax.set_title(f"{scenario}: {title}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_online_error_plot(
    path: Path,
    *,
    scenario: str,
    errors: dict[str, torch.Tensor],
    labels: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, values in errors.items():
        cpu_values = values.detach().cpu()
        axes[0].plot(cpu_values.numpy(), label=labels.get(name, name), linewidth=1.1)
        axes[1].plot(
            torch.cumsum(cpu_values, dim=0).numpy(),
            label=labels.get(name, name),
            linewidth=1.1,
        )
    axes[0].set_ylabel("one-step error")
    axes[1].set_ylabel("cumulative error")
    axes[1].set_xlabel("online update step")
    axes[0].set_title(scenario)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_online_component_error_plot(
    path: Path,
    *,
    scenario: str,
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    labels: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_cpu = target.detach().cpu()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    plot_specs = [
        (0, "|delta x error|"),
        (1, "|delta xdot error|"),
    ]
    positive_values: list[float] = []
    for name, prediction in predictions.items():
        prediction_cpu = prediction.detach().cpu()
        component_error = (prediction_cpu - target_cpu).abs()
        norm_error = torch.linalg.norm(prediction_cpu - target_cpu, dim=-1)
        for axis_index, (component_index, _) in enumerate(plot_specs):
            values = component_error[:, component_index]
            positive_values.extend(float(value) for value in values if value > 0.0)
            axes[axis_index].plot(
                values.numpy(),
                label=labels.get(name, name),
                linewidth=1.0,
            )
        positive_values.extend(float(value) for value in norm_error if value > 0.0)
        axes[2].plot(norm_error.numpy(), label=labels.get(name, name), linewidth=1.0)

    for axis, (_, label) in zip(axes[:2], plot_specs):
        axis.set_ylabel(label)
    axes[2].set_ylabel("norm error")
    axes[2].set_xlabel("online update step")
    if positive_values and max(positive_values) / max(min(positive_values), 1e-12) > 1_000.0:
        for axis in axes:
            axis.set_yscale("log")
    axes[0].set_title(scenario)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_recursive_horizon_plot(
    path: Path,
    *,
    scenario: str,
    rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(
        {
            int(str(key).removeprefix("recursive_k").removesuffix("_final_step_error_mean"))
            for row in rows
            for key in row
            if str(key).startswith("recursive_k")
            and str(key).endswith("_final_step_error_mean")
        }
    )
    if not horizons:
        path.write_text("")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    positive_values: list[float] = []
    for row in rows:
        label = str(row["label"])
        final_values = [float(row[f"recursive_k{k}_final_step_error_mean"]) for k in horizons]
        accum_values = [float(row[f"recursive_k{k}_accumulated_error_mean"]) for k in horizons]
        positive_values.extend(value for value in final_values + accum_values if value > 0.0)
        axes[0].plot(horizons, final_values, marker="o", label=label, linewidth=1.0)
        axes[1].plot(horizons, accum_values, marker="o", label=label, linewidth=1.0)
    if positive_values and max(positive_values) / max(min(positive_values), 1e-12) > 1_000.0:
        for axis in axes:
            axis.set_yscale("log")
    axes[0].set_title("final-step error")
    axes[1].set_title("accumulated error")
    for axis in axes:
        axis.set_xlabel("recursive rollout horizon")
        axis.set_xticks(horizons)
    axes[0].set_ylabel("error")
    axes[0].legend(fontsize=8)
    fig.suptitle(scenario)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_phase_plot(
    path: Path,
    *,
    scenario: str,
    trajectory: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    labels: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_states = trajectory["states"].detach().cpu()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(true_states[:, 0], true_states[:, 1], label="true", linewidth=2)
    for name, deltas in predictions.items():
        if name == "zero_delta":
            continue
        states = [true_states[0]]
        for delta in deltas.detach().cpu():
            states.append(states[-1] + delta)
        state_tensor = torch.stack(states)
        ax.plot(
            state_tensor[:, 0],
            state_tensor[:, 1],
            label=labels.get(name, name),
            linewidth=1,
            alpha=0.75,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("xdot")
    ax.set_title(scenario)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_rollout_snapshot_plot(
    path: Path,
    *,
    scenario: str,
    trajectory: dict[str, torch.Tensor],
    methods: dict[str, torch.nn.Module],
    coefficient_histories: dict[str, torch.Tensor],
    initial_coefficients: dict[str, torch.Tensor],
    labels: dict[str, str],
    rollout_horizon: int = 100,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_states = trajectory["states"].detach().cpu()
    max_start = max(0, true_states.shape[0] - 2)
    starts = _unique_indices([0, max_start // 3, 2 * max_start // 3])
    xlim, ylim = _phase_limits(true_states)
    fig, axes = plt.subplots(1, len(starts), figsize=(5 * len(starts), 4), squeeze=False)
    for col_index, start in enumerate(starts):
        ax = axes[0][col_index]
        horizon = min(rollout_horizon, true_states.shape[0] - start - 1)
        true_segment = true_states[start : start + horizon + 1]
        ax.plot(true_segment[:, 0], true_segment[:, 1], label="true", linewidth=2)
        for name, model in methods.items():
            rollout = rollout_fixed_coefficients(
                model,
                trajectory=trajectory,
                coefficient_history=coefficient_histories[name],
                initial_coefficients=initial_coefficients.get(name),
                start=start,
                horizon=horizon,
            )
            ax.plot(
                rollout[:, 0],
                rollout[:, 1],
                label=labels.get(name, name),
                linewidth=1.0,
                alpha=0.8,
            )
        zero_rollout = rollout_fixed_coefficients(
            None,
            trajectory=trajectory,
            coefficient_history=None,
            start=start,
            horizon=horizon,
        )
        ax.plot(
            zero_rollout[:, 0],
            zero_rollout[:, 1],
            label=labels.get("zero_delta", "zero delta"),
            linewidth=1.0,
            alpha=0.8,
            linestyle="--",
        )
        ax.scatter(true_segment[0, 0], true_segment[0, 1], s=16, color="black", zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("xdot")
        ax.set_title(f"start {start}, horizon {horizon}")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{scenario}: fixed-coefficient recursive rollouts")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streamplot_comparison(
    path: Path,
    *,
    scenario: str,
    mu: float,
    trajectory: dict[str, torch.Tensor],
    methods: dict[str, torch.nn.Module],
    coefficient_histories: dict[str, torch.Tensor],
    labels: dict[str, str],
    dt: float,
    grid_size: int = 31,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_states = trajectory["states"].detach().cpu()
    xlim, ylim = _phase_limits(true_states)
    device = trajectory["states"].device
    dtype = trajectory["states"].dtype
    grid_x, grid_y, points = _phase_grid(
        xlim=xlim,
        ylim=ylim,
        grid_size=grid_size,
        device=device,
        dtype=dtype,
    )

    plot_items: list[tuple[str, torch.Tensor]] = [
        ("true", true_vanderpol_vector_field(points, mu=mu)),
    ]
    for name, model in methods.items():
        coefficients = coefficient_histories[name][-1].to(device=device, dtype=dtype).unsqueeze(0)
        plot_items.append(
            (
                labels.get(name, name),
                predict_vector_field(model, points=points, coefficients=coefficients, dt=dt),
            )
        )

    fig, axes = plt.subplots(1, len(plot_items), figsize=(4 * len(plot_items), 4), squeeze=False)
    for col_index, (title, vectors) in enumerate(plot_items):
        ax = axes[0][col_index]
        stream_u, stream_v = _stream_components(vectors, grid_size=grid_size)
        ax.streamplot(grid_x.numpy(), grid_y.numpy(), stream_u.numpy(), stream_v.numpy(), density=1.0)
        ax.plot(true_states[:, 0], true_states[:, 1], color="black", linewidth=1.0, alpha=0.55)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("xdot")
        ax.set_title(title)
    fig.suptitle(f"{scenario}: final adapted vector fields")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_basis_streamplots(
    path: Path,
    *,
    model: torch.nn.Module,
    title: str,
    dt: float,
    grid_size: int = 25,
    max_basis: int = 8,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    xlim = (-3.0, 3.0)
    ylim = (-6.0, 6.0)
    grid_x, grid_y, points = _phase_grid(
        xlim=xlim,
        ylim=ylim,
        grid_size=grid_size,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        dt_values = torch.full((1, points.shape[0]), dt, dtype=dtype, device=device)
        features = model(RuntimeInput(points.reshape(1, -1, 2), dt_values)).squeeze(0)
        vectors = features / dt

    n_plots = min(int(model.n_basis), max_basis)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for basis_index in range(n_rows * n_cols):
        ax = axes[basis_index // n_cols][basis_index % n_cols]
        if basis_index >= n_plots:
            ax.axis("off")
            continue
        stream_u, stream_v = _stream_components(vectors[:, :, basis_index], grid_size=grid_size)
        ax.streamplot(grid_x.numpy(), grid_y.numpy(), stream_u.numpy(), stream_v.numpy(), density=0.9)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("xdot")
        ax.set_title(f"basis {basis_index}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def rollout_fixed_coefficients(
    predictor: torch.nn.Module | None,
    *,
    trajectory: dict[str, torch.Tensor],
    coefficient_history: torch.Tensor | None,
    start: int,
    horizon: int,
    initial_coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    states = trajectory["states"]
    dt = trajectory["dt"]
    current = states[start].clone()
    rollout = [current.detach().cpu()]
    coefficients = None
    if predictor is not None:
        assert coefficient_history is not None
        coefficients = _coefficients_for_rollout_start(
            coefficient_history=coefficient_history,
            initial_coefficients=initial_coefficients,
            start=start,
            device=states.device,
            dtype=states.dtype,
        )
    for offset in range(horizon):
        index = start + offset
        if predictor is None:
            delta = torch.zeros_like(current)
        else:
            assert coefficients is not None
            features = predictor(
                RuntimeInput(
                    current.reshape(1, 1, -1),
                    dt[index : index + 1].reshape(1, 1),
                )
            )
            delta = linear_predict(features, coefficients).squeeze(0).squeeze(0)
        current = current + delta
        rollout.append(current.detach().cpu())
        if not torch.isfinite(current).all():
            break
    return torch.stack(rollout)


@torch.no_grad()
def predict_vector_field(
    model: torch.nn.Module,
    *,
    points: torch.Tensor,
    coefficients: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    dt_values = torch.full((1, points.shape[0]), dt, dtype=points.dtype, device=points.device)
    features = model(RuntimeInput(points.reshape(1, -1, 2), dt_values))
    deltas = linear_predict(features, coefficients).squeeze(0)
    return deltas / dt


def true_vanderpol_vector_field(points: torch.Tensor, *, mu: float) -> torch.Tensor:
    return torch.stack(
        (
            points[:, 1],
            mu * (1.0 - points[:, 0].square()) * points[:, 1] - points[:, 0],
        ),
        dim=-1,
    )


def write_training_loss_plot(
    path: Path,
    *,
    train_histories: dict[str, dict[str, object]],
    labels: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_validation = any(history.get("validation_losses") for history in train_histories.values())
    fig, axes = plt.subplots(2 if has_validation else 1, 1, figsize=(7, 6 if has_validation else 4))
    if not isinstance(axes, list) and not hasattr(axes, "__len__"):
        axes = [axes]
    train_ax = axes[0]
    positive_values: list[float] = []
    validation_positive: list[float] = []
    for name, history in train_histories.items():
        losses = [float(value) for value in history.get("losses", [])]
        if not losses:
            continue
        positive_values.extend(value for value in losses if value > 0.0)
        train_ax.plot(losses, label=labels.get(name, name), linewidth=1.1)
        if has_validation:
            validation_steps = [int(value) for value in history.get("validation_steps", [])]
            validation_losses = [float(value) for value in history.get("validation_losses", [])]
            validation_positive.extend(value for value in validation_losses if value > 0.0)
            axes[1].plot(
                validation_steps,
                validation_losses,
                marker="o",
                label=labels.get(name, name),
                linewidth=1.1,
            )
    if positive_values and max(positive_values) / max(min(positive_values), 1e-12) > 100.0:
        train_ax.set_yscale("log")
    train_ax.set_xlabel("training step")
    train_ax.set_ylabel("query MSE")
    train_ax.set_title("basis training losses")
    train_ax.legend()
    if has_validation:
        if validation_positive and max(validation_positive) / max(min(validation_positive), 1e-12) > 100.0:
            axes[1].set_yscale("log")
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("validation query MSE")
        axes[1].set_title("validation losses")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_metric_bar_grid(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = []
    for row in rows:
        scenario = str(row["scenario"])
        if scenario not in scenarios:
            scenarios.append(scenario)
    metrics = [
        ("first_25_mean_error", "first 25 online steps"),
        ("adaptation_samples_to_25pct_improvement", "samples to 25% adaptation"),
        ("mean_error", "all online steps"),
        ("recursive_k10_accumulated_error_mean", "recursive k=10 accumulated"),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        len(scenarios),
        figsize=(6 * len(scenarios), 2.7 * len(metrics)),
    )
    if len(scenarios) == 1:
        axes = [[ax] for ax in axes]
    for row_index, (metric, title) in enumerate(metrics):
        for col_index, scenario in enumerate(scenarios):
            ax = axes[row_index][col_index]
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            scenario_rows = sorted(scenario_rows, key=lambda row: float(row[metric]))
            labels = [str(row["label"]) for row in scenario_rows]
            raw_values = [float(row[metric]) for row in scenario_rows]
            finite_values = [
                value for value in raw_values if torch.isfinite(torch.tensor(value)) and value > 0.0
            ]
            if finite_values:
                max_finite = max(finite_values)
                min_finite = min(finite_values)
                plot_values = [
                    value
                    if torch.isfinite(torch.tensor(value)) and value > 0.0
                    else max_finite * 10.0
                    for value in raw_values
                ]
            else:
                min_finite = 1e-12
                max_finite = 1.0
                plot_values = [1.0 for _ in raw_values]
            use_log = max_finite / max(min_finite, 1e-12) > 100.0
            ax.bar(range(len(labels)), plot_values)
            if use_log:
                ax.set_yscale("log")
                ax.set_ylabel("error (log scale)")
            if any(not torch.isfinite(torch.tensor(value)) for value in raw_values):
                ax.text(
                    0.98,
                    0.95,
                    "non-finite clipped",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
            ax.set_title(f"{scenario}: {title}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _unique_indices(indices: list[int]) -> list[int]:
    unique: list[int] = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    return unique


def _phase_limits(states: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = states[torch.isfinite(states).all(dim=-1)]
    if finite.numel() == 0:
        return (-3.0, 3.0), (-6.0, 6.0)
    mins = finite.min(dim=0).values
    maxs = finite.max(dim=0).values
    spans = torch.clamp(maxs - mins, min=1e-6)
    pads = torch.maximum(0.15 * spans, torch.tensor([0.5, 0.75], dtype=states.dtype))
    xlim = (float(mins[0] - pads[0]), float(maxs[0] + pads[0]))
    ylim = (float(mins[1] - pads[1]), float(maxs[1] + pads[1]))
    return xlim, ylim


def _phase_grid(
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_values = torch.linspace(xlim[0], xlim[1], grid_size, dtype=dtype, device=device)
    y_values = torch.linspace(ylim[0], ylim[1], grid_size, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(y_values, x_values, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return x_values.detach().cpu(), y_values.detach().cpu(), points


def _stream_components(vectors: torch.Tensor, *, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_vectors = torch.nan_to_num(
        vectors.detach().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    u = cpu_vectors[:, 0].reshape(grid_size, grid_size)
    v = cpu_vectors[:, 1].reshape(grid_size, grid_size)
    return u, v


def _coefficients_for_rollout_start(
    *,
    coefficient_history: torch.Tensor,
    initial_coefficients: torch.Tensor | None,
    start: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if start <= 0:
        coefficients = (
            torch.zeros_like(coefficient_history[0])
            if initial_coefficients is None
            else initial_coefficients
        )
    else:
        coefficients = coefficient_history[min(start - 1, coefficient_history.shape[0] - 1)]
    return coefficients.to(device=device, dtype=dtype).unsqueeze(0)


def _true_vanderpol_delta(x: torch.Tensor, *, mu: float, dt: torch.Tensor) -> torch.Tensor:
    def derivative(state: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                state[..., 1],
                mu * (1.0 - state[..., 0].square()) * state[..., 1] - state[..., 0],
            ),
            dim=-1,
        )

    k1 = derivative(x)
    k2 = derivative(x + 0.5 * dt * k1)
    k3 = derivative(x + 0.5 * dt * k2)
    k4 = derivative(x + dt * k3)
    return dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _rk4_delta(field: torch.nn.Module, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    k1 = field(x)
    k2 = field(x + 0.5 * dt.unsqueeze(-1) * k1)
    k3 = field(x + 0.5 * dt.unsqueeze(-1) * k2)
    k4 = field(x + dt.unsqueeze(-1) * k3)
    return dt.unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _concat_state_dt(inputs: RuntimeInput) -> torch.Tensor:
    xs = _as_batched_state(inputs.xs)
    dt = _as_batched_dt(inputs.dt, xs)
    return torch.cat((xs, dt.unsqueeze(-1)), dim=-1)


def _as_batched_state(xs: object) -> torch.Tensor:
    tensor = xs if isinstance(xs, torch.Tensor) else torch.as_tensor(xs)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _as_batched_dt(dt: object, xs: torch.Tensor) -> torch.Tensor:
    tensor = dt if isinstance(dt, torch.Tensor) else torch.as_tensor(dt, device=xs.device)
    tensor = tensor.to(dtype=xs.dtype, device=xs.device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _scenario_name(mu: float, train_mus: tuple[float, ...]) -> str:
    lower = min(train_mus)
    upper = max(train_mus)
    if lower <= mu <= upper:
        return f"interpolation_mu_{mu:g}"
    return f"extrapolation_mu_{mu:g}"


def _prefix_mean(values: torch.Tensor, n: int) -> float:
    return float(values[: min(n, values.shape[0])].mean())


def _suffix_mean(values: torch.Tensor, n: int) -> float:
    return float(values[-min(n, values.shape[0]) :].mean())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _is_float_like(value: object) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value
