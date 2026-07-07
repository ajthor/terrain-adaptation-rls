"""Function Encoder RLS streaming diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config
from terrain_adaptation_rls.data.load_data import load_scenes
from terrain_adaptation_rls.evaluation.diagnostic_plots import integrate_planar_deltas
from terrain_adaptation_rls.methods.runtime import (
    FunctionEncoderBasisProvider,
    Observation,
    RuntimeInput,
    TorchCoefficientMethod,
)
from terrain_adaptation_rls.models.function_encoder import create_model
from terrain_adaptation_rls.training.supervised import scene_supervised_batch


@dataclass(frozen=True)
class FERLSArtifacts:
    """Summary of a completed FE-RLS streaming diagnostic run."""

    artifact_dir: Path
    summary: dict[str, object]


def run_fe_rls_streaming_diagnostic(
    *,
    train_run_dir: str | Path,
    artifact_dir: str | Path,
    scene: str,
    device: torch.device | str = "cpu",
    max_points: int = 512,
    start_index: int = 0,
    n_example_points: int | None = None,
    forgetting_factor: float = 0.95,
    initial_covariance: float = 1_000.0,
    measurement_noise: float = 1e-6,
) -> FERLSArtifacts:
    """Evaluate FE-RLS prediction-before-update behavior on one scene."""

    train_run_dir = Path(train_run_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(train_run_dir / "resolved_config.json")
    if config.platform is None:
        raise ValueError("Training config must include platform")

    device = torch.device(device)
    model = load_trained_function_encoder(config, train_run_dir, device=device)
    inputs, targets = load_scenes([scene], config.platform)[scene]
    xs, dt, target, time = scene_streaming_tensors(
        inputs=inputs,
        targets=targets,
        start_index=start_index,
        max_points=max_points,
        device=device,
    )

    n_examples = n_example_points or int(config.training.get("n_example_points", 128))
    offline_example_batch = scene_supervised_batch(
        inputs=inputs,
        targets=targets,
        n_example_points=n_examples,
        max_query_points=1,
        device=device,
        seed=config.seed,
    )
    _, _, _, example_xs, example_dt, example_ys = offline_example_batch
    offline_coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)
    offline_prediction = model((xs, dt), coefficients=offline_coefficients)

    method = TorchCoefficientMethod(
        FunctionEncoderBasisProvider(model),
        update_rule="rls",
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        device=device,
    )

    state = method.initial_state()
    predictions: list[torch.Tensor] = []
    coefficient_after_updates: list[torch.Tensor] = []

    with torch.no_grad():
        for idx in range(xs.shape[1]):
            step_inputs = RuntimeInput(
                xs=xs[:, idx : idx + 1],
                dt=dt[:, idx : idx + 1],
            )
            step_target = target[:, idx : idx + 1]
            prediction = method.predict(state, step_inputs)
            state = method.update(
                state,
                Observation(
                    inputs=step_inputs,
                    target=step_target,
                    time=float(time[idx].detach().cpu()),
                ),
            )
            predictions.append(prediction.squeeze(0).squeeze(0).detach().cpu())
            coefficient_after_updates.append(state.coefficients.squeeze(0).detach().cpu())

    rls_prediction = torch.stack(predictions).unsqueeze(0)
    coefficients = torch.stack(coefficient_after_updates)
    target_cpu = target.detach().cpu()
    dt_cpu = dt.detach().cpu()
    time_cpu = time.detach().cpu()
    offline_prediction_cpu = offline_prediction.detach().cpu()
    zero_prediction = torch.zeros_like(rls_prediction)

    summary = summarize_streaming_predictions(
        target=target_cpu,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction_cpu,
        zero_prediction=zero_prediction,
        coefficients=coefficients,
        scene=scene,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
    )
    write_fe_rls_artifacts(
        artifact_dir,
        scene=scene,
        time=time_cpu,
        dt=dt_cpu,
        target=target_cpu,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction_cpu,
        zero_prediction=zero_prediction,
        coefficients=coefficients,
        summary=summary,
    )
    return FERLSArtifacts(artifact_dir=artifact_dir, summary=summary)


def load_trained_function_encoder(
    config: ExperimentConfig,
    train_run_dir: Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Load an FE checkpoint using the resolved training config."""

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)
    state = torch.load(train_run_dir / "function_encoder_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def scene_streaming_tensors(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    start_index: int,
    max_points: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice one contiguous streaming sequence into FE input/target tensors."""

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same row count")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    stop_index = min(start_index + max_points, inputs.shape[0])
    if start_index >= stop_index:
        raise ValueError("requested streaming slice is empty")

    selected_inputs = inputs[start_index:stop_index]
    selected_targets = targets[start_index:stop_index]
    xs = selected_inputs[:, 1:]
    dt = selected_targets[:, 0] - selected_inputs[:, 0]
    target = selected_targets[:, 1:] - xs[:, :6]
    time = selected_inputs[:, 0] - selected_inputs[0, 0]

    device = torch.device(device)
    return (
        xs.unsqueeze(0).to(device),
        dt.unsqueeze(0).to(device),
        target.unsqueeze(0).to(device),
        time.to(device),
    )


def summarize_streaming_predictions(
    *,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
    zero_prediction: torch.Tensor,
    coefficients: torch.Tensor,
    scene: str,
    forgetting_factor: float,
    initial_covariance: float,
    measurement_noise: float,
) -> dict[str, object]:
    """Compute scalar FE-RLS diagnostic metrics."""

    target_delta = target.squeeze(0)
    rls_delta = rls_prediction.squeeze(0)
    offline_delta = offline_prediction.squeeze(0)
    zero_delta = zero_prediction.squeeze(0)
    rls_error = torch.linalg.norm(rls_delta - target_delta, dim=-1)
    offline_error = torch.linalg.norm(offline_delta - target_delta, dim=-1)
    zero_error = torch.linalg.norm(zero_delta - target_delta, dim=-1)
    coefficient_norm = torch.linalg.norm(coefficients, dim=-1)

    return {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "rls": _prediction_summary(target_delta, rls_delta, rls_error, zero_error),
        "offline_scene_coefficients": _prediction_summary(
            target_delta,
            offline_delta,
            offline_error,
            zero_error,
        ),
        "zero_delta_baseline": {
            "mean_error": float(zero_error.mean()),
            "final_accumulated_error": float(zero_error.sum()),
            "mse": float(torch.nn.functional.mse_loss(zero_delta, target_delta)),
        },
        "coefficients": {
            "final_norm": float(coefficient_norm[-1]),
            "mean_norm": float(coefficient_norm.mean()),
            "max_norm": float(coefficient_norm.max()),
            "final": [float(value) for value in coefficients[-1]],
        },
        "rls_parameters": {
            "forgetting_factor": forgetting_factor,
            "initial_covariance": initial_covariance,
            "measurement_noise": measurement_noise,
        },
    }


def write_fe_rls_artifacts(
    artifact_dir: Path,
    *,
    scene: str,
    time: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
    zero_prediction: torch.Tensor,
    coefficients: torch.Tensor,
    summary: dict[str, object],
) -> None:
    """Write FE-RLS CSV, JSON, and plot artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_streaming_predictions_csv(
        artifact_dir / "streaming_predictions.csv",
        time=time,
        target=target,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction,
    )
    write_streaming_error_plot(
        artifact_dir / "streaming_error.png",
        scene=scene,
        time=time,
        target=target,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction,
        zero_prediction=zero_prediction,
    )
    write_coefficient_plot(
        artifact_dir / "rls_coefficients.png",
        scene=scene,
        time=time,
        coefficients=coefficients,
    )
    write_streaming_components_plot(
        artifact_dir / "streaming_components.png",
        scene=scene,
        time=time,
        target=target,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction,
    )
    write_streaming_trajectory_plot(
        artifact_dir / "streaming_trajectory.png",
        scene=scene,
        target=target,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction,
    )
    write_delta_scale_plot(
        artifact_dir / "streaming_delta_scale.png",
        scene=scene,
        time=time,
        target=target,
        rls_prediction=rls_prediction,
        offline_prediction=offline_prediction,
    )


def write_streaming_predictions_csv(
    path: Path,
    *,
    time: torch.Tensor,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
) -> None:
    """Write target and prediction values for each streaming step."""

    target = target.squeeze(0)
    rls_prediction = rls_prediction.squeeze(0)
    offline_prediction = offline_prediction.squeeze(0)
    rls_error = torch.linalg.norm(rls_prediction - target, dim=-1)
    offline_error = torch.linalg.norm(offline_prediction - target, dim=-1)

    with path.open("w", newline="") as f:
        fieldnames = ["index", "time", "rls_error", "offline_error"]
        fieldnames += [f"target_{idx}" for idx in range(target.shape[-1])]
        fieldnames += [f"rls_prediction_{idx}" for idx in range(target.shape[-1])]
        fieldnames += [f"offline_prediction_{idx}" for idx in range(target.shape[-1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(target.shape[0]):
            row = {
                "index": idx,
                "time": float(time[idx]),
                "rls_error": float(rls_error[idx]),
                "offline_error": float(offline_error[idx]),
            }
            row.update({f"target_{dim}": float(target[idx, dim]) for dim in range(target.shape[-1])})
            row.update(
                {
                    f"rls_prediction_{dim}": float(rls_prediction[idx, dim])
                    for dim in range(target.shape[-1])
                }
            )
            row.update(
                {
                    f"offline_prediction_{dim}": float(offline_prediction[idx, dim])
                    for dim in range(target.shape[-1])
                }
            )
            writer.writerow(row)


def write_streaming_error_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
    zero_prediction: torch.Tensor,
) -> None:
    """Write per-step and accumulated streaming error plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = target.squeeze(0)
    rls_prediction = rls_prediction.squeeze(0)
    offline_prediction = offline_prediction.squeeze(0)
    zero_prediction = zero_prediction.squeeze(0)
    rls_error = torch.linalg.norm(rls_prediction - target, dim=-1)
    offline_error = torch.linalg.norm(offline_prediction - target, dim=-1)
    zero_error = torch.linalg.norm(zero_prediction - target, dim=-1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time, rls_error, label="FE-RLS")
    axes[0].plot(time, offline_error, label="offline FE")
    axes[0].plot(time, zero_error, label="zero delta", alpha=0.5)
    axes[0].set_ylabel("error norm")
    axes[0].legend()

    axes[1].plot(time, torch.cumsum(rls_error, dim=0), label="FE-RLS")
    axes[1].plot(time, torch.cumsum(offline_error, dim=0), label="offline FE")
    axes[1].plot(time, torch.cumsum(zero_error, dim=0), label="zero delta", alpha=0.5)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("accumulated error")
    axes[1].legend()
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_coefficient_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    coefficients: torch.Tensor,
) -> None:
    """Write coefficient trajectories and coefficient norm."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeff_norm = torch.linalg.norm(coefficients, dim=-1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for idx in range(coefficients.shape[-1]):
        axes[0].plot(time, coefficients[:, idx], linewidth=1.0, label=f"c{idx}")
    axes[0].set_ylabel("coefficient")
    if coefficients.shape[-1] <= 10:
        axes[0].legend(ncol=2)
    axes[1].plot(time, coeff_norm)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("coefficient norm")
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_components_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
) -> None:
    """Write per-dimension target and prediction traces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = target.squeeze(0)
    rls_prediction = rls_prediction.squeeze(0)
    offline_prediction = offline_prediction.squeeze(0)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    for dim, ax in enumerate(axes.ravel()):
        ax.plot(time, target[:, dim], label="target", linewidth=1.2)
        ax.plot(time, rls_prediction[:, dim], label="FE-RLS", linewidth=1.0)
        ax.plot(time, offline_prediction[:, dim], label="offline FE", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"dim {dim}")
    axes.ravel()[0].legend()
    axes.ravel()[-1].set_xlabel("relative time [s]")
    axes.ravel()[-2].set_xlabel("relative time [s]")
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_trajectory_plot(
    path: Path,
    *,
    scene: str,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
) -> None:
    """Write integrated local trajectory snapshot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_pose = integrate_planar_deltas(target.squeeze(0))
    rls_pose = integrate_planar_deltas(rls_prediction.squeeze(0))
    offline_pose = integrate_planar_deltas(offline_prediction.squeeze(0))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(target_pose[:, 0], target_pose[:, 1], label="target", linewidth=1.5)
    ax.plot(rls_pose[:, 0], rls_pose[:, 1], label="FE-RLS", linewidth=1.2)
    ax.plot(offline_pose[:, 0], offline_pose[:, 1], label="offline FE", linewidth=1.0)
    ax.scatter(target_pose[0, 0], target_pose[0, 1], s=20, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("integrated local x")
    ax.set_ylabel("integrated local y")
    ax.set_title(scene)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_delta_scale_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    rls_prediction: torch.Tensor,
    offline_prediction: torch.Tensor,
) -> None:
    """Write delta norm and cumulative planar distance for FE-RLS."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = target.squeeze(0)
    rls_prediction = rls_prediction.squeeze(0)
    offline_prediction = offline_prediction.squeeze(0)
    target_norm = torch.linalg.norm(target, dim=-1)
    rls_norm = torch.linalg.norm(rls_prediction, dim=-1)
    offline_norm = torch.linalg.norm(offline_prediction, dim=-1)
    target_planar = torch.cumsum(torch.linalg.norm(target[:, :2], dim=-1), dim=0)
    rls_planar = torch.cumsum(torch.linalg.norm(rls_prediction[:, :2], dim=-1), dim=0)
    offline_planar = torch.cumsum(torch.linalg.norm(offline_prediction[:, :2], dim=-1), dim=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time, target_norm, label="target")
    axes[0].plot(time, rls_norm, label="FE-RLS")
    axes[0].plot(time, offline_norm, label="offline FE")
    axes[0].set_ylabel("delta norm")
    axes[0].legend()
    axes[1].plot(time, target_planar, label="target")
    axes[1].plot(time, rls_planar, label="FE-RLS")
    axes[1].plot(time, offline_planar, label="offline FE")
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("cumulative planar delta")
    axes[1].legend()
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _prediction_summary(
    target: torch.Tensor,
    prediction: torch.Tensor,
    error: torch.Tensor,
    zero_error: torch.Tensor,
) -> dict[str, float]:
    return {
        "mean_error": float(error.mean()),
        "final_accumulated_error": float(error.sum()),
        "mean_error_to_zero_delta_ratio": _safe_ratio(
            float(error.mean()),
            float(zero_error.mean()),
        ),
        "mse": float(torch.nn.functional.mse_loss(prediction, target)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
