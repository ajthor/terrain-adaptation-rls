"""Standalone weak-form Function Encoder sanity experiment on Van der Pol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch


class ToyMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        *,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        activation_module = _activation_module(activation)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            activation_module,
            torch.nn.Linear(hidden_size, hidden_size),
            _activation_module(activation),
            torch.nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VectorFieldBasis(torch.nn.Module):
    """Independent vector-field basis functions ``g_i(x)``."""

    def __init__(self, *, n_basis: int, hidden_size: int, activation: str = "tanh") -> None:
        super().__init__()
        self.n_basis = int(n_basis)
        self.basis_functions = torch.nn.ModuleList(
            [ToyMLP(2, 2, hidden_size, activation=activation) for _ in range(self.n_basis)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = [basis(x) for basis in self.basis_functions]
        return torch.stack(values, dim=-1)


def run_vdp_weak_fe_experiment(
    *,
    artifact_dir: str | Path,
    device: torch.device | str = "cpu",
    epochs: int = 300,
    seed: int = 123,
    noise: float = 0.02,
    dt: float = 0.05,
    steps: int = 160,
    rollout_steps: int = 300,
    n_basis: int = 4,
    hidden_size: int = 64,
    activation: str = "tanh",
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    ridge: float = 1e-4,
    norm_weight: float = 1e-3,
    coeff_weight: float = 1e-5,
    gradient_clip: float = 1.0,
    window: int = 41,
    powers: Sequence[int] = (4, 6, 8, 10),
    example_starts: Sequence[int] = (0, 15, 30, 45),
    query_starts: Sequence[int] = (60, 75, 90, 105),
    eval_mus: Sequence[float] = (0.6, 1.0, 1.7, 2.4),
    eval_example_starts: Sequence[int] = (0, 15, 30, 45, 60, 75),
    write_plots: bool = True,
) -> dict[str, object]:
    """Train and evaluate the known-good weak-form FE toy setup."""

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dtype = torch.float32
    torch.manual_seed(seed)

    model = VectorFieldBasis(n_basis=n_basis, hidden_size=hidden_size, activation=activation).to(
        device=device,
        dtype=dtype,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    weak_losses: list[float] = []
    norm_losses: list[float] = []
    coeff_losses: list[float] = []
    for _ in range(epochs):
        _, _, observed = batch_trajectories(
            batch_size,
            dt=dt,
            steps=steps,
            noise=noise,
            device=device,
            dtype=dtype,
        )
        basis = model(observed)
        context_design, context_target = weak_system(
            observed,
            basis,
            dt=dt,
            starts=example_starts,
            window=window,
            powers=powers,
        )
        coefficients, gram = solve_coefficients(
            context_design,
            context_target,
            regularization=ridge,
        )
        query_design, query_target = weak_system(
            observed,
            basis,
            dt=dt,
            starts=query_starts,
            window=window,
            powers=powers,
        )
        prediction = torch.einsum("bnk,bk->bn", query_design, coefficients)
        weak_loss = torch.nn.functional.mse_loss(prediction, query_target)
        gram_diag = torch.diagonal(gram.mean(dim=0))
        norm_loss = ((gram_diag / gram_diag.detach().mean().clamp_min(1e-6) - 1.0) ** 2).mean()
        coeff_loss = (coefficients**2).mean()
        loss = weak_loss + norm_weight * norm_loss + coeff_weight * coeff_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        weak_losses.append(float(weak_loss.detach().cpu()))
        norm_losses.append(float(norm_loss.detach().cpu()))
        coeff_losses.append(float(coeff_loss.detach().cpu()))

    eval_rows = evaluate_rollouts(
        model,
        dt=dt,
        steps=steps,
        rollout_steps=rollout_steps,
        noise=noise,
        mus=eval_mus,
        example_starts=eval_example_starts,
        window=window,
        powers=powers,
        ridge=ridge,
        device=device,
        dtype=dtype,
    )
    summary: dict[str, object] = {
        "epochs": epochs,
        "seed": seed,
        "noise": noise,
        "dt": dt,
        "steps": steps,
        "rollout_steps": rollout_steps,
        "n_basis": n_basis,
        "hidden_size": hidden_size,
        "activation": activation,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ridge": ridge,
        "norm_weight": norm_weight,
        "coeff_weight": coeff_weight,
        "gradient_clip": gradient_clip,
        "window": window,
        "powers": list(powers),
        "example_starts": list(example_starts),
        "query_starts": list(query_starts),
        "eval_example_starts": list(eval_example_starts),
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "weak_losses": weak_losses,
        "norm_losses": norm_losses,
        "coeff_losses": coeff_losses,
        "eval_rows": eval_rows,
    }
    (artifact_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_eval_csv(artifact_path / "rollout_summary.csv", eval_rows)
    torch.save(model.state_dict(), artifact_path / "weak_fe_model.pth")
    if write_plots:
        write_training_plot(artifact_path / "training_losses.png", summary)
        write_rollout_plot(artifact_path / "vdp_weak_fe_rollouts.png", eval_rows)
    return summary


def _activation_module(name: str) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "tanh":
        return torch.nn.Tanh()
    if normalized == "relu":
        return torch.nn.ReLU()
    raise ValueError(f"unsupported activation: {name}")


def vdp_rhs(x: torch.Tensor, *, mu: torch.Tensor) -> torch.Tensor:
    mu = mu.squeeze(-1)
    return torch.stack(
        [x[..., 1], mu * (1.0 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]],
        dim=-1,
    )


def rk4_step(func, x: torch.Tensor, dt: float, **kwargs) -> torch.Tensor:
    k1 = func(x, **kwargs)
    k2 = func(x + 0.5 * dt * k1, **kwargs)
    k3 = func(x + 0.5 * dt * k2, **kwargs)
    k4 = func(x + dt * k3, **kwargs)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_vdp(
    *,
    mu: torch.Tensor,
    y0: torch.Tensor,
    dt: float,
    steps: int,
) -> torch.Tensor:
    xs = [y0]
    current = y0
    for _ in range(steps):
        current = rk4_step(vdp_rhs, current, dt, mu=mu)
        xs.append(current)
    return torch.stack(xs, dim=1)


def polynomial_test(tau: torch.Tensor, power: int) -> torch.Tensor:
    return (1.0 - tau**2) ** power


def polynomial_test_derivative(tau: torch.Tensor, power: int) -> torch.Tensor:
    return -2.0 * power * tau * (1.0 - tau**2) ** (power - 1)


def trapz_weights(n: int, *, dt: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones(n, device=device, dtype=dtype) * dt
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def weak_system(
    x: torch.Tensor,
    basis: torch.Tensor,
    *,
    dt: float,
    starts: Sequence[int],
    window: int,
    powers: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flattened weak system rows from fixed windows."""

    device = x.device
    dtype = x.dtype
    tau = torch.linspace(-1.0, 1.0, window, device=device, dtype=dtype)
    horizon = dt * (window - 1)
    weights = trapz_weights(window, dt=dt, device=device, dtype=dtype)

    design_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for start in starts:
        x_window = x[:, start : start + window]
        basis_window = basis[:, start : start + window]
        for power in powers:
            phi = polynomial_test(tau, power)
            dphi_dt = polynomial_test_derivative(tau, power) * (2.0 / horizon)
            design = torch.einsum("t,btdk->bdk", weights * phi, basis_window)
            target = -torch.einsum("t,btd->bd", weights * dphi_dt, x_window)
            design_rows.append(design)
            target_rows.append(target)

    design = torch.stack(design_rows, dim=1).flatten(1, 2)
    target = torch.stack(target_rows, dim=1).flatten(1, 2)
    return design, target


def solve_coefficients(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gram = torch.einsum("bnk,bnl->bkl", design, design)
    rhs = torch.einsum("bnk,bn->bk", design, target)
    eye = torch.eye(gram.shape[-1], device=design.device, dtype=design.dtype)
    return torch.linalg.solve(gram + regularization * eye, rhs), gram


def batch_trajectories(
    batch_size: int,
    *,
    dt: float,
    steps: int,
    noise: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.5, 2.5)
    y0 = torch.empty(batch_size, 2, device=device, dtype=dtype).uniform_(-3.0, 3.0)
    clean = simulate_vdp(mu=mu, y0=y0, dt=dt, steps=steps)
    scale = clean.std(dim=1, keepdim=True).clamp_min(1e-6)
    observed = clean + noise * scale * torch.randn_like(clean)
    return mu.squeeze(-1), clean, observed


def rollout_model(
    model: VectorFieldBasis,
    *,
    y0: torch.Tensor,
    coefficients: torch.Tensor,
    dt: float,
    steps: int,
) -> torch.Tensor:
    states = [y0]
    current = y0
    for _ in range(steps):

        def rhs(z: torch.Tensor, *, coefficients: torch.Tensor) -> torch.Tensor:
            basis = model(z.unsqueeze(1)).squeeze(1)
            return torch.einsum("bdk,bk->bd", basis, coefficients)

        current = rk4_step(rhs, current, dt, coefficients=coefficients)
        states.append(current)
    return torch.stack(states, dim=1)


@torch.no_grad()
def evaluate_rollouts(
    model: VectorFieldBasis,
    *,
    dt: float,
    steps: int,
    rollout_steps: int,
    noise: float,
    mus: Sequence[float],
    example_starts: Sequence[int],
    window: int,
    powers: Sequence[int],
    ridge: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mu_value in mus:
        mu = torch.tensor([[mu_value]], device=device, dtype=dtype)
        y0 = torch.tensor([[2.0, 0.0]], device=device, dtype=dtype)
        clean_fit = simulate_vdp(mu=mu, y0=y0, dt=dt, steps=steps)
        scale = clean_fit.std(dim=1, keepdim=True).clamp_min(1e-6)
        observed = clean_fit + noise * scale * torch.randn_like(clean_fit)
        basis = model(observed)
        design, target = weak_system(
            observed,
            basis,
            dt=dt,
            starts=example_starts,
            window=window,
            powers=powers,
        )
        coefficients, gram = solve_coefficients(design, target, regularization=ridge)
        prediction = rollout_model(
            model,
            y0=y0,
            coefficients=coefficients,
            dt=dt,
            steps=rollout_steps,
        )
        clean = simulate_vdp(mu=mu, y0=y0, dt=dt, steps=rollout_steps)
        error = torch.linalg.norm(prediction - clean, dim=-1)
        rows.append(
            {
                "mu": float(mu_value),
                "mean_error": float(error.mean().cpu()),
                "final_error": float(error[:, -1].mean().cpu()),
                "max_error": float(error.max().cpu()),
                "trajectory_rmse": float(torch.sqrt(torch.mean((prediction - clean) ** 2)).cpu()),
                "accumulated_error": float(error.sum().cpu()),
                "coefficient_norm": float(torch.linalg.norm(coefficients).cpu()),
                "gram_condition": float(torch.linalg.cond(gram.squeeze(0)).cpu()),
                "clean": clean.squeeze(0).detach().cpu().tolist(),
                "prediction": prediction.squeeze(0).detach().cpu().tolist(),
                "observed": observed.squeeze(0).detach().cpu().tolist(),
            }
        )
    return rows


def write_eval_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    fieldnames = [
        "mu",
        "mean_error",
        "final_error",
        "max_error",
        "trajectory_rmse",
        "accumulated_error",
        "coefficient_norm",
        "gram_condition",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_training_plot(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (
        ("losses", "total"),
        ("weak_losses", "weak"),
        ("norm_losses", "gram norm"),
        ("coeff_losses", "coeff"),
    ):
        values = summary.get(key, [])
        if values:
            ax.plot(values, label=label, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_rollout_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, 2)
    for row_index, row in enumerate(rows):
        clean = torch.tensor(row["clean"])
        prediction = torch.tensor(row["prediction"])
        observed = torch.tensor(row["observed"])
        error = torch.linalg.norm(prediction - clean, dim=-1)
        time = torch.arange(clean.shape[0]) * 1.0

        ax = axes[row_index, 0]
        ax.plot(clean[:, 0], clean[:, 1], label="true", linewidth=2.0)
        ax.plot(prediction[:, 0], prediction[:, 1], label="weak FE")
        ax.scatter(observed[::10, 0], observed[::10, 1], s=8, alpha=0.25, label="fit data")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"mu={float(row['mu']):.2f}")
        if row_index == 0:
            ax.legend(fontsize=8)

        ax = axes[row_index, 1]
        ax.plot(time, error)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e2)
        ax.set_title("rollout error")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
