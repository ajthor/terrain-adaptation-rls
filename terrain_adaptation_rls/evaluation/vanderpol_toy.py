"""Van der Pol toy-system comparisons for zero-start online adaptation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from terrain_adaptation_rls.estimators.linear import RLSState, linear_predict, rls_update
from terrain_adaptation_rls.estimators.linear import solve_ridge_coefficients
from terrain_adaptation_rls.methods.runtime import RuntimeInput


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

    def forward(self, inputs: RuntimeInput) -> torch.Tensor:
        z = _concat_state_dt(inputs)
        features = self.network(z)
        return features.reshape(*z.shape[:-1], 2, self.n_basis)


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
    dt: float = 0.02,
    train_mus: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
    test_mus: tuple[float, ...] = (1.25, 3.0),
    train_trajectories_per_mu: int = 8,
    train_trajectory_steps: int = 160,
    eval_steps: int = 500,
    forgetting_factor: float = 0.98,
    initial_covariance: float = 100.0,
    measurement_noise: float = 1e-5,
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
    methods: dict[str, torch.nn.Module] = {
        "fe_ode_rls": FEODEBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
        "fe_mlp_rls": FEIncrementBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
        "neuralfly_rls": NeuralFlyToyBasis(n_basis=n_basis, hidden_size=hidden_size).to(device),
    }
    labels = {
        "fe_ode_rls": "FE-ODE basis RLS",
        "fe_mlp_rls": "FE-MLP basis RLS",
        "neuralfly_rls": "NeuralFly-style RLS",
        "zero_delta": "zero delta",
    }

    train_histories = {
        name: train_toy_basis(
            model,
            train_data=train_data,
            steps=train_steps,
            batch_size=batch_size,
            n_example_points=n_example_points,
            n_query_points=n_query_points,
            ridge=ridge,
            learning_rate=learning_rate,
            seed=seed + index,
            device=device,
        )
        for index, (name, model) in enumerate(methods.items())
    }

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
        scenario_rows: list[dict[str, object]] = []

        for name, model in methods.items():
            result = evaluate_zero_start_rls(
                model,
                trajectory=trajectory,
                forgetting_factor=forgetting_factor,
                initial_covariance=initial_covariance,
                measurement_noise=measurement_noise,
                recursive_horizons=(1, 5, 10, 25),
            )
            row = {
                "scenario": scenario,
                "mu": mu,
                "method": name,
                "label": labels[name],
                **result["metrics"],
            }
            rows.append(row)
            scenario_rows.append(row)
            scenario_predictions[name] = result["predictions"]
            scenario_errors[name] = result["errors"]

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

        write_online_error_plot(
            artifact_path / f"{scenario}_online_error.png",
            scenario=scenario,
            errors=scenario_errors,
            labels=labels,
        )
        write_phase_plot(
            artifact_path / f"{scenario}_phase_trajectory.png",
            scenario=scenario,
            trajectory=trajectory,
            predictions=scenario_predictions,
            labels=labels,
        )
        scenarios[scenario] = {
            "mu": mu,
            "rows": scenario_rows,
        }

    write_rows_csv(artifact_path / "method_summary.csv", rows)
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
        "dt": dt,
        "train_mus": list(train_mus),
        "test_mus": list(test_mus),
        "forgetting_factor": forgetting_factor,
        "initial_covariance": initial_covariance,
        "measurement_noise": measurement_noise,
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
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses: list[float] = []
    for _ in range(steps):
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
        loss = torch.nn.functional.mse_loss(prediction, batch["query_y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }


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


@torch.no_grad()
def evaluate_zero_start_rls(
    model: torch.nn.Module,
    *,
    trajectory: dict[str, torch.Tensor],
    forgetting_factor: float,
    initial_covariance: float,
    measurement_noise: float,
    recursive_horizons: tuple[int, ...],
) -> dict[str, object]:
    xs = trajectory["xs"]
    dt = trajectory["dt"]
    target = trajectory["deltas"]
    n_coeff = int(model.n_basis)
    state = RLSState(
        coefficients=torch.zeros(1, n_coeff, dtype=xs.dtype, device=xs.device),
        covariance=initial_covariance
        * torch.eye(n_coeff, dtype=xs.dtype, device=xs.device).unsqueeze(0),
    )
    predictions: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    for index in range(xs.shape[0]):
        features = model(
            RuntimeInput(
                xs[index : index + 1].unsqueeze(0),
                dt[index : index + 1].unsqueeze(0),
            )
        )
        prediction = linear_predict(features, state.coefficients).squeeze(0).squeeze(0)
        state = rls_update(
            state,
            features.squeeze(1),
            target[index : index + 1],
            forgetting_factor=forgetting_factor,
            measurement_noise=measurement_noise,
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
        "metrics": metrics,
    }


def summarize_online_errors(
    *,
    errors: torch.Tensor,
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    zero_error = torch.linalg.norm(target, dim=-1)
    return {
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


@torch.no_grad()
def summarize_recursive_errors(
    *,
    predictor: torch.nn.Module | None,
    trajectory: dict[str, torch.Tensor],
    coefficient_history: torch.Tensor | None,
    horizons: tuple[int, ...],
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
            coefficients = (
                torch.zeros_like(coefficient_history[0]).unsqueeze(0)
                if start == 0
                else coefficient_history[start - 1].unsqueeze(0)
            )
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


def write_training_loss_plot(
    path: Path,
    *,
    train_histories: dict[str, dict[str, object]],
    labels: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    positive_values: list[float] = []
    for name, history in train_histories.items():
        losses = [float(value) for value in history.get("losses", [])]
        if not losses:
            continue
        positive_values.extend(value for value in losses if value > 0.0)
        ax.plot(losses, label=labels.get(name, name), linewidth=1.1)
    if positive_values and max(positive_values) / max(min(positive_values), 1e-12) > 100.0:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("query MSE")
    ax.set_title("basis training losses")
    ax.legend()
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
        ("mean_error", "all online steps"),
        ("recursive_k10_accumulated_error_mean", "recursive k=10 accumulated"),
    ]
    fig, axes = plt.subplots(len(metrics), len(scenarios), figsize=(6 * len(scenarios), 8))
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
