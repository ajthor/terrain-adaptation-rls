"""Diagnostic plots for supervised dynamics runs."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import torch


STATE_LABELS = ("x", "y", "yaw", "x velocity", "y velocity", "yaw rate")


def write_supervised_diagnostics(
    artifact_dir: str | Path,
    *,
    model: torch.nn.Module,
    family: str,
    batch: tuple[torch.Tensor, ...],
    prediction: torch.Tensor,
    scene: str,
) -> None:
    """Write extra debug plots for one supervised validation batch."""

    artifact_dir = Path(artifact_dir)
    xs, dt, target, *_ = batch
    error = torch.linalg.norm(prediction - target, dim=-1)

    write_conditioning_summary(
        artifact_dir / "conditioning_summary.json",
        model=model,
        family=family,
        batch=batch,
        prediction=prediction,
        scene=scene,
    )
    write_trajectory_summary(
        artifact_dir / "trajectory_summary.json",
        target=target,
        prediction=prediction,
        dt=dt,
        scene=scene,
    )
    write_delta_scale_plot(
        artifact_dir / "validation_delta_scale.png",
        target=target,
        prediction=prediction,
        dt=dt,
        scene=scene,
    )
    write_error_histogram(artifact_dir / "validation_error_histogram.png", error)
    write_trajectory_snapshot(
        artifact_dir / "validation_trajectory_snapshot.png",
        target=target,
        prediction=prediction,
        scene=scene,
    )
    if family == "function_encoder":
        write_function_encoder_phase_plots(
            artifact_dir,
            model=model,
            batch=batch,
            scene=scene,
            state_x_dim=3,
            state_y_dim=5,
            output_x_dim=3,
            output_y_dim=5,
        )


def integrate_planar_deltas(deltas: torch.Tensor) -> torch.Tensor:
    """Integrate ``[dx_body, dy_body, dyaw, ...]`` deltas into planar poses."""

    if deltas.ndim != 2 or deltas.shape[-1] < 3:
        raise ValueError(f"deltas must have shape [steps, >=3], got {tuple(deltas.shape)}")

    poses = deltas.new_zeros(deltas.shape[0] + 1, 3)
    for idx, delta in enumerate(deltas):
        yaw = poses[idx, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_body = delta[0]
        dy_body = delta[1]
        poses[idx + 1, 0] = poses[idx, 0] + cos_yaw * dx_body - sin_yaw * dy_body
        poses[idx + 1, 1] = poses[idx, 1] + sin_yaw * dx_body + cos_yaw * dy_body
        poses[idx + 1, 2] = poses[idx, 2] + delta[2]
    return poses


def summarize_trajectory_scales(
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dt: torch.Tensor | None = None,
    scene: str | None = None,
) -> dict[str, object]:
    """Summarize prediction-vs-target scale for one contiguous trajectory batch."""

    target_delta = _single_trajectory(target)
    prediction_delta = _single_trajectory(prediction)
    if target_delta.shape != prediction_delta.shape:
        raise ValueError(
            "target and prediction shapes must match: "
            f"{tuple(target_delta.shape)} != {tuple(prediction_delta.shape)}"
        )

    error_delta = prediction_delta - target_delta
    zero_delta = torch.zeros_like(target_delta)
    zero_error_delta = zero_delta - target_delta
    target_pose = integrate_planar_deltas(target_delta)
    prediction_pose = integrate_planar_deltas(prediction_delta)
    zero_pose = integrate_planar_deltas(zero_delta)
    target_planar_norm = torch.linalg.norm(target_delta[:, :2], dim=-1)
    prediction_planar_norm = torch.linalg.norm(prediction_delta[:, :2], dim=-1)
    error_mean = _mean_norm(error_delta)
    zero_error_mean = _mean_norm(zero_error_delta)

    summary: dict[str, object] = {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "dt": _dt_summary(dt),
        "delta_norms": {
            "target_mean": _mean_norm(target_delta),
            "prediction_mean": _mean_norm(prediction_delta),
            "error_mean": error_mean,
            "zero_delta_error_mean": zero_error_mean,
            "prediction_error_to_zero_delta_error_ratio": _safe_ratio(
                error_mean,
                zero_error_mean,
            ),
            "prediction_to_target_ratio": _safe_ratio(
                _mean_norm(prediction_delta),
                _mean_norm(target_delta),
            ),
            "target_planar_mean": float(target_planar_norm.mean()),
            "prediction_planar_mean": float(prediction_planar_norm.mean()),
            "prediction_to_target_planar_ratio": _safe_ratio(
                float(prediction_planar_norm.mean()),
                float(target_planar_norm.mean()),
            ),
        },
        "trajectory": {
            "target": _pose_summary(target_pose),
            "prediction": _pose_summary(prediction_pose),
            "zero_delta_baseline": _pose_summary(zero_pose),
        },
        "per_dimension": _per_dimension_summary(target_delta, prediction_delta),
        "flags": [],
    }

    ratio = summary["delta_norms"]["prediction_to_target_planar_ratio"]
    if ratio is not None and ratio < 0.1:
        summary["flags"].append("prediction_planar_deltas_are_less_than_10_percent_of_target")
    target_path = summary["trajectory"]["target"]["path_length"]
    prediction_path = summary["trajectory"]["prediction"]["path_length"]
    path_ratio = _safe_ratio(prediction_path, target_path)
    summary["trajectory"]["prediction_to_target_path_length_ratio"] = path_ratio
    if path_ratio is not None and path_ratio < 0.1:
        summary["flags"].append("prediction_path_length_is_less_than_10_percent_of_target")
    error_ratio = summary["delta_norms"]["prediction_error_to_zero_delta_error_ratio"]
    if error_ratio is not None and 0.95 <= error_ratio <= 1.05:
        summary["flags"].append("prediction_error_is_within_5_percent_of_zero_delta_baseline")
    return summary


def write_trajectory_summary(
    path: str | Path,
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dt: torch.Tensor | None,
    scene: str,
) -> None:
    """Write scale and displacement diagnostics for a trajectory comparison."""

    summary = summarize_trajectory_scales(
        target=target,
        prediction=prediction,
        dt=dt,
        scene=scene,
    )
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def write_conditioning_summary(
    path: str | Path,
    *,
    model: torch.nn.Module,
    family: str,
    batch: tuple[torch.Tensor, ...],
    prediction: torch.Tensor,
    scene: str,
) -> None:
    """Write scale diagnostics for the query and FE conditioning examples."""

    summary = summarize_conditioning(
        model=model,
        family=family,
        batch=batch,
        prediction=prediction,
        scene=scene,
    )
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def summarize_conditioning(
    *,
    model: torch.nn.Module,
    family: str,
    batch: tuple[torch.Tensor, ...],
    prediction: torch.Tensor,
    scene: str | None = None,
) -> dict[str, object]:
    """Summarize the scale of examples, queries, coefficients, and predictions."""

    xs, dt, target, example_xs, example_dt, example_ys = batch
    zero_prediction = torch.zeros_like(prediction)
    example_target_norm = _batched_mean_norm(example_ys)
    query_target_norm = _batched_mean_norm(target)
    prediction_norm = _batched_mean_norm(prediction)
    summary: dict[str, object] = {
        "scene": scene,
        "family": family,
        "examples": {
            "n_points": int(example_ys.shape[1]),
            "target_norm_mean": example_target_norm,
            "target_abs_mean": _mean_abs(example_ys),
            "dt": _batched_time_summary(example_dt),
        },
        "query": {
            "n_points": int(target.shape[1]),
            "target_norm_mean": query_target_norm,
            "target_abs_mean": _mean_abs(target),
            "prediction_norm_mean": prediction_norm,
            "prediction_abs_mean": _mean_abs(prediction),
            "prediction_to_target_norm_ratio": _safe_ratio(
                prediction_norm,
                query_target_norm,
            ),
            "mse": _mse(prediction, target),
            "zero_delta_mse": _mse(zero_prediction, target),
            "mse_to_zero_delta_mse_ratio": _safe_ratio(
                _mse(prediction, target),
                _mse(zero_prediction, target),
            ),
            "dt": _batched_time_summary(dt),
        },
        "example_to_query_target_norm_ratio": _safe_ratio(
            example_target_norm,
            query_target_norm,
        ),
        "flags": [],
    }

    example_query_ratio = summary["example_to_query_target_norm_ratio"]
    if example_query_ratio is not None and example_query_ratio < 0.1:
        summary["flags"].append("conditioning_examples_are_less_than_10_percent_of_query_scale")
    if example_query_ratio is not None and example_query_ratio > 10.0:
        summary["flags"].append("conditioning_examples_are_more_than_10x_query_scale")

    if family == "function_encoder":
        coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
        basis = model.basis_functions((example_xs, example_dt))
        example_prediction = model((example_xs, example_dt), coefficients=coefficients)
        coefficient_norms = torch.linalg.norm(coefficients.detach(), dim=-1).cpu()
        summary["function_encoder"] = {
            "coefficient_norms": [float(value) for value in coefficient_norms],
            "coefficient_abs_mean": _mean_abs(coefficients),
            "coefficient_abs_max": _max_abs(coefficients),
            "basis_example_abs_mean": _mean_abs(basis),
            "basis_example_abs_max": _max_abs(basis),
            "example_prediction_norm_mean": _batched_mean_norm(example_prediction),
            "example_mse": _mse(example_prediction, example_ys),
        }

    return summary


def write_delta_scale_plot(
    path: str | Path,
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dt: torch.Tensor,
    scene: str,
) -> None:
    """Plot delta norms and cumulative planar distance for target/prediction."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = _single_trajectory(target)
    prediction_delta = _single_trajectory(prediction)
    time = torch.cumsum(_single_time(dt), dim=0)
    target_delta_norm = torch.linalg.norm(target_delta, dim=-1)
    prediction_delta_norm = torch.linalg.norm(prediction_delta, dim=-1)
    target_planar_distance = torch.cumsum(torch.linalg.norm(target_delta[:, :2], dim=-1), dim=0)
    prediction_planar_distance = torch.cumsum(
        torch.linalg.norm(prediction_delta[:, :2], dim=-1),
        dim=0,
    )

    path = Path(path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time.cpu(), target_delta_norm.cpu(), label="target")
    axes[0].plot(time.cpu(), prediction_delta_norm.cpu(), label="prediction")
    axes[0].set_ylabel("delta norm")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(time.cpu(), target_planar_distance.cpu(), label="target")
    axes[1].plot(time.cpu(), prediction_planar_distance.cpu(), label="prediction")
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("cumulative planar delta")
    axes[1].legend()
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_error_histogram(path: str | Path, error: torch.Tensor) -> None:
    """Write a histogram of one-step prediction error norms."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(error.detach().cpu().flatten(), bins=32)
    ax.set_xlabel("prediction error norm")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _single_trajectory(value: torch.Tensor) -> torch.Tensor:
    value = value.squeeze(0).detach().cpu()
    if value.ndim != 2:
        raise ValueError(f"expected one trajectory with shape [steps, dim], got {tuple(value.shape)}")
    return value


def _single_time(dt: torch.Tensor) -> torch.Tensor:
    value = dt.squeeze(0).detach().cpu()
    if value.ndim != 1:
        raise ValueError(f"expected dt with shape [steps], got {tuple(value.shape)}")
    return value


def _dt_summary(dt: torch.Tensor | None) -> dict[str, float] | None:
    if dt is None:
        return None
    values = _single_time(dt)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "total": float(values.sum()),
    }


def _batched_time_summary(dt: torch.Tensor) -> dict[str, float]:
    values = dt.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _mean_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value, dim=-1).mean())


def _batched_mean_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value.detach(), dim=-1).mean().cpu())


def _mean_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().mean().cpu())


def _max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu())


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(prediction.detach(), target.detach()).cpu())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def _pose_summary(poses: torch.Tensor) -> dict[str, float]:
    planar_steps = poses[1:, :2] - poses[:-1, :2]
    final_xy = poses[-1, :2]
    return {
        "final_x": float(poses[-1, 0]),
        "final_y": float(poses[-1, 1]),
        "final_yaw": float(poses[-1, 2]),
        "final_displacement": float(torch.linalg.norm(final_xy)),
        "path_length": float(torch.linalg.norm(planar_steps, dim=-1).sum()),
    }


def _per_dimension_summary(
    target_delta: torch.Tensor,
    prediction_delta: torch.Tensor,
) -> list[dict[str, float | str | None]]:
    summaries: list[dict[str, float | str | None]] = []
    error_delta = prediction_delta - target_delta
    for dim in range(target_delta.shape[-1]):
        target_values = target_delta[:, dim]
        prediction_values = prediction_delta[:, dim]
        target_mean_abs = float(target_values.abs().mean())
        prediction_mean_abs = float(prediction_values.abs().mean())
        label = STATE_LABELS[dim] if dim < len(STATE_LABELS) else f"dim {dim}"
        summaries.append(
            {
                "dim": dim,
                "label": label,
                "target_mean_abs": target_mean_abs,
                "prediction_mean_abs": prediction_mean_abs,
                "prediction_to_target_mean_abs_ratio": _safe_ratio(
                    prediction_mean_abs,
                    target_mean_abs,
                ),
                "target_rms": float(torch.sqrt(torch.mean(target_values.square()))),
                "prediction_rms": float(torch.sqrt(torch.mean(prediction_values.square()))),
                "error_rms": float(torch.sqrt(torch.mean(error_delta[:, dim].square()))),
                "target_min": float(target_values.min()),
                "target_max": float(target_values.max()),
                "prediction_min": float(prediction_values.min()),
                "prediction_max": float(prediction_values.max()),
            }
        )
    return summaries


def write_trajectory_snapshot(
    path: str | Path,
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    scene: str,
) -> None:
    """Write an integrated local-displacement target vs prediction snapshot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_pose = integrate_planar_deltas(target.squeeze(0).detach().cpu())
    prediction_pose = integrate_planar_deltas(prediction.squeeze(0).detach().cpu())

    path = Path(path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(target_pose[:, 0], target_pose[:, 1], label="target", linewidth=1.5)
    ax.plot(prediction_pose[:, 0], prediction_pose[:, 1], label="prediction", linewidth=1.5)
    ax.scatter(target_pose[0, 0], target_pose[0, 1], s=20, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("integrated local x")
    ax.set_ylabel("integrated local y")
    ax.set_title(scene)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def write_function_encoder_phase_plots(
    artifact_dir: str | Path,
    *,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    scene: str,
    state_x_dim: int,
    state_y_dim: int,
    output_x_dim: int,
    output_y_dim: int,
    resolution: int = 25,
    max_basis_plots: int = 8,
) -> None:
    """Write FE coefficient and phase-portrait diagnostics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact_dir = Path(artifact_dir)
    xs, dt, _, example_xs, example_dt, example_ys = batch
    coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
    grid_xs, grid_dt, x_values, y_values = make_phase_grid(
        xs,
        dt,
        state_x_dim=state_x_dim,
        state_y_dim=state_y_dim,
        resolution=resolution,
    )

    basis = model.basis_functions((grid_xs, grid_dt))
    prediction = model((grid_xs, grid_dt), coefficients=coefficients)
    dt_scale = grid_dt.squeeze(0).median().clamp_min(1e-6)

    write_coefficient_plot(
        artifact_dir / "fe_coefficients.png",
        coefficients=coefficients.squeeze(0).detach().cpu(),
        scene=scene,
    )
    write_streamplot(
        artifact_dir / "phase_streamplot.png",
        x_values=x_values,
        y_values=y_values,
        vector_x=(prediction[0, :, output_x_dim] / dt_scale).detach().cpu(),
        vector_y=(prediction[0, :, output_y_dim] / dt_scale).detach().cpu(),
        title=f"{scene}: adapted FE field",
        xlabel=STATE_LABELS[state_x_dim],
        ylabel=STATE_LABELS[state_y_dim],
    )
    write_basis_streamplots(
        artifact_dir / "basis_streamplots.png",
        basis=basis.detach().cpu(),
        x_values=x_values,
        y_values=y_values,
        output_x_dim=output_x_dim,
        output_y_dim=output_y_dim,
        dt_scale=float(dt_scale.detach().cpu()),
        max_basis_plots=max_basis_plots,
        xlabel=STATE_LABELS[state_x_dim],
        ylabel=STATE_LABELS[state_y_dim],
    )


def make_phase_grid(
    xs: torch.Tensor,
    dt: torch.Tensor,
    *,
    state_x_dim: int,
    state_y_dim: int,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a grid over two state dimensions around observed validation data."""

    if xs.ndim != 3:
        raise ValueError(f"xs must have shape [batch, points, features], got {tuple(xs.shape)}")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    reference = xs[0].detach()
    base = reference.median(dim=0).values
    x_min, x_max = quantile_range(reference[:, state_x_dim])
    y_min, y_max = quantile_range(reference[:, state_y_dim])
    x_values = torch.linspace(x_min, x_max, resolution, device=xs.device, dtype=xs.dtype)
    y_values = torch.linspace(y_min, y_max, resolution, device=xs.device, dtype=xs.dtype)
    mesh_x, mesh_y = torch.meshgrid(x_values, y_values, indexing="xy")

    grid = base.repeat(resolution * resolution, 1)
    grid[:, state_x_dim] = mesh_x.reshape(-1)
    grid[:, state_y_dim] = mesh_y.reshape(-1)
    grid_dt = dt.detach().median().expand(resolution * resolution)
    return grid.unsqueeze(0), grid_dt.unsqueeze(0), x_values.detach().cpu(), y_values.detach().cpu()


def quantile_range(values: torch.Tensor) -> tuple[float, float]:
    """Return a non-degenerate central range for plotting."""

    lower = float(torch.quantile(values.detach().float().cpu(), 0.02))
    upper = float(torch.quantile(values.detach().float().cpu(), 0.98))
    if upper <= lower:
        center = 0.5 * (lower + upper)
        lower = center - 1.0
        upper = center + 1.0
    margin = 0.05 * (upper - lower)
    return lower - margin, upper + margin


def write_coefficient_plot(path: str | Path, *, coefficients: torch.Tensor, scene: str) -> None:
    """Write a bar plot of FE coefficients computed from examples."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    indices = torch.arange(coefficients.numel())
    ax.bar(indices.numpy(), coefficients.numpy())
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("basis index")
    ax.set_ylabel("coefficient")
    ax.set_title(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streamplot(
    path: str | Path,
    *,
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    vector_x: torch.Tensor,
    vector_y: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Write one streamplot from flattened vector values."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolution = x_values.numel()
    u = vector_x.reshape(resolution, resolution).numpy()
    v = vector_y.reshape(resolution, resolution).numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.streamplot(x_values.numpy(), y_values.numpy(), u, v, density=1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_basis_streamplots(
    path: str | Path,
    *,
    basis: torch.Tensor,
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    output_x_dim: int,
    output_y_dim: int,
    dt_scale: float,
    max_basis_plots: int,
    xlabel: str,
    ylabel: str,
) -> None:
    """Write a grid of FE basis-function phase streamplots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_basis = basis.shape[-1]
    n_plots = min(n_basis, max_basis_plots)
    n_cols = min(4, n_plots)
    n_rows = ceil(n_plots / n_cols)
    resolution = x_values.numel()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n_plots:
            ax.axis("off")
            continue
        u = (basis[0, :, output_x_dim, idx] / dt_scale).reshape(resolution, resolution).numpy()
        v = (basis[0, :, output_y_dim, idx] / dt_scale).reshape(resolution, resolution).numpy()
        ax.streamplot(x_values.numpy(), y_values.numpy(), u, v, density=0.9)
        ax.set_title(f"basis {idx}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
