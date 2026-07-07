"""Online baseline comparisons on one Phoenix scene."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from terrain_adaptation_rls.configuration import load_config
from terrain_adaptation_rls.data.load_data import load_scenes
from terrain_adaptation_rls.estimators.linear import linear_predict, solve_ridge_coefficients
from terrain_adaptation_rls.evaluation.diagnostic_plots import integrate_planar_deltas
from terrain_adaptation_rls.evaluation.fe_rls import (
    load_trained_function_encoder,
    scene_streaming_tensors,
)
from terrain_adaptation_rls.methods.runtime import (
    FunctionEncoderBasisProvider,
    LinearBasisProvider,
    Observation,
    RuntimeInput,
    TorchCoefficientMethod,
)
from terrain_adaptation_rls.training.neuralfly import (
    load_neuralfly_style_basis,
    predict_neuralfly_style_batch,
)
from terrain_adaptation_rls.training.supervised import scene_supervised_batch


METHOD_LABELS = {
    "fe_rls": "FE-RLS",
    "fe_kalman": "FE-Kalman",
    "fe_sgd": "FE-SGD",
    "fe_window_ls": "FE-window LS",
    "offline_fe": "offline FE",
    "neuralfly_rls": "NeuralFly-style RLS",
    "offline_neuralfly": "offline NeuralFly-style",
    "static_node": "static NODE",
    "linear_rls": "linear RLS",
    "offline_linear": "offline linear",
    "zero_delta": "zero delta",
}


@dataclass(frozen=True)
class OnlineBaselineArtifacts:
    """Summary of a completed online baseline comparison."""

    artifact_dir: Path
    summary: dict[str, object]


def run_online_baseline_comparison(
    *,
    fe_run_dir: str | Path,
    artifact_dir: str | Path,
    scene: str,
    neuralfly_run_dir: str | Path | None = None,
    node_run_dir: str | Path | None = None,
    device: torch.device | str = "cpu",
    max_points: int = 512,
    start_index: int = 0,
    n_example_points: int | None = None,
    forgetting_factor: float = 0.95,
    initial_covariance: float = 1_000.0,
    measurement_noise: float = 1e-6,
    include_fe_variants: bool = True,
    kalman_process_noise: float = 0.0,
    fe_sgd_learning_rate: float = 1.0,
    fe_sgd_momentum: float = 0.0,
    fe_sgd_weight_decay: float = 0.0,
    fe_window_size: int = 100,
    fe_window_ridge: float = 1e-6,
    linear_include_bias: bool = True,
) -> OnlineBaselineArtifacts:
    """Compare trained and online update baselines on one held-out scene."""

    fe_run_dir = Path(fe_run_dir)
    neuralfly_run_path = None if neuralfly_run_dir is None else Path(neuralfly_run_dir)
    node_run_path = None if node_run_dir is None else Path(node_run_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fe_config = load_config(fe_run_dir / "resolved_config.json")
    if fe_config.platform is None:
        raise ValueError("FE training config must include platform")

    device = torch.device(device)
    inputs, targets = load_scenes([scene], fe_config.platform)[scene]
    xs, dt, target, time = scene_streaming_tensors(
        inputs=inputs,
        targets=targets,
        start_index=start_index,
        max_points=max_points,
        device=device,
    )

    n_examples = n_example_points or int(fe_config.training.get("n_example_points", 128))
    example_batch = scene_supervised_batch(
        inputs=inputs,
        targets=targets,
        n_example_points=n_examples,
        max_query_points=1,
        device=device,
        seed=fe_config.seed,
    )
    _, _, _, example_xs, example_dt, example_ys = example_batch

    predictions: dict[str, torch.Tensor] = {}
    coefficient_histories: dict[str, torch.Tensor] = {}

    fe_model = load_trained_function_encoder(fe_config, fe_run_dir, device=device)
    fe_coefficients, _ = fe_model.compute_coefficients((example_xs, example_dt), example_ys)
    predictions["offline_fe"] = fe_model((xs, dt), coefficients=fe_coefficients).detach().cpu()

    fe_method = TorchCoefficientMethod(
        FunctionEncoderBasisProvider(fe_model),
        update_rule="rls",
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        device=device,
    )
    predictions["fe_rls"], coefficient_histories["fe_rls"] = stream_runtime_method(
        fe_method,
        xs=xs,
        dt=dt,
        target=target,
        time=time,
    )

    if include_fe_variants:
        fe_provider = FunctionEncoderBasisProvider(fe_model)
        fe_variant_methods = {
            "fe_kalman": TorchCoefficientMethod(
                fe_provider,
                update_rule="kalman",
                output_dim=6,
                initial_covariance=initial_covariance,
                measurement_noise=measurement_noise,
                process_noise=kalman_process_noise,
                device=device,
            ),
            "fe_sgd": TorchCoefficientMethod(
                fe_provider,
                update_rule="sgd",
                output_dim=6,
                learning_rate=fe_sgd_learning_rate,
                momentum=fe_sgd_momentum,
                weight_decay=fe_sgd_weight_decay,
                device=device,
            ),
            "fe_window_ls": TorchCoefficientMethod(
                fe_provider,
                update_rule="window_ls",
                output_dim=6,
                window_size=fe_window_size,
                ridge=fe_window_ridge,
                device=device,
            ),
        }
        for name, method in fe_variant_methods.items():
            predictions[name], coefficient_histories[name] = stream_runtime_method(
                method,
                xs=xs,
                dt=dt,
                target=target,
                time=time,
            )

    if neuralfly_run_path is not None:
        neuralfly_config = load_config(neuralfly_run_path / "resolved_config.json")
        neuralfly_model = load_neuralfly_style_basis(
            neuralfly_config,
            neuralfly_run_path,
            device=device,
        )
        ridge = float(neuralfly_config.model.get("ridge", 1e-6))
        predictions["offline_neuralfly"] = predict_neuralfly_style_batch(
            neuralfly_model,
            (xs, dt, target, example_xs, example_dt, example_ys),
            ridge=ridge,
        ).detach().cpu()
        neuralfly_method = TorchCoefficientMethod(
            neuralfly_model,
            update_rule="rls",
            output_dim=6,
            forgetting_factor=forgetting_factor,
            initial_covariance=initial_covariance,
            measurement_noise=measurement_noise,
            device=device,
        )
        (
            predictions["neuralfly_rls"],
            coefficient_histories["neuralfly_rls"],
        ) = stream_runtime_method(
            neuralfly_method,
            xs=xs,
            dt=dt,
            target=target,
            time=time,
        )

    if node_run_path is not None:
        node_config = load_config(node_run_path / "resolved_config.json")
        node_model = load_trained_neural_ode(node_config, node_run_path, device=device)
        predictions["static_node"] = node_model((xs, dt)).detach().cpu()

    linear_provider = LinearBasisProvider(
        input_dim=9,
        output_dim=6,
        include_bias=linear_include_bias,
    )
    linear_example_features = linear_provider(RuntimeInput(example_xs, example_dt))
    linear_coefficients = solve_ridge_coefficients(
        linear_example_features,
        example_ys,
        ridge=1e-6,
    )
    linear_query_features = linear_provider(RuntimeInput(xs, dt))
    predictions["offline_linear"] = linear_predict(
        linear_query_features,
        linear_coefficients,
    ).detach().cpu()

    linear_method = TorchCoefficientMethod(
        linear_provider,
        update_rule="rls",
        output_dim=6,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        device=device,
    )
    predictions["linear_rls"], coefficient_histories["linear_rls"] = stream_runtime_method(
        linear_method,
        xs=xs,
        dt=dt,
        target=target,
        time=time,
    )

    target_cpu = target.detach().cpu()
    dt_cpu = dt.detach().cpu()
    time_cpu = time.detach().cpu()
    predictions["zero_delta"] = torch.zeros_like(target_cpu)

    summary = summarize_baseline_predictions(
        target=target_cpu,
        predictions=predictions,
        coefficient_histories=coefficient_histories,
        scene=scene,
        forgetting_factor=forgetting_factor,
        initial_covariance=initial_covariance,
        measurement_noise=measurement_noise,
        start_index=start_index,
        n_example_points=n_examples,
        include_fe_variants=include_fe_variants,
        kalman_process_noise=kalman_process_noise,
        fe_sgd_learning_rate=fe_sgd_learning_rate,
        fe_sgd_momentum=fe_sgd_momentum,
        fe_sgd_weight_decay=fe_sgd_weight_decay,
        fe_window_size=fe_window_size,
        fe_window_ridge=fe_window_ridge,
        linear_include_bias=linear_include_bias,
    )
    write_online_baseline_artifacts(
        artifact_dir,
        scene=scene,
        time=time_cpu,
        dt=dt_cpu,
        target=target_cpu,
        predictions=predictions,
        coefficient_histories=coefficient_histories,
        summary=summary,
    )
    return OnlineBaselineArtifacts(artifact_dir=artifact_dir, summary=summary)


def load_trained_neural_ode(
    config: object,
    train_run_dir: str | Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained static Neural ODE from a run directory."""

    from terrain_adaptation_rls.models.neural_ode import create_model

    n_basis = int(config.model.get("n_basis", 8))
    hidden_size = int(config.model.get("hidden_size", 128))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)
    state = torch.load(Path(train_run_dir) / "neural_ode_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def stream_runtime_method(
    method: TorchCoefficientMethod,
    *,
    xs: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream one trajectory through a predict-before-update runtime method."""

    state = method.initial_state()
    predictions: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []

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
        coefficients.append(state.coefficients.squeeze(0).detach().cpu())

    return torch.stack(predictions).unsqueeze(0), torch.stack(coefficients)


def summarize_baseline_predictions(
    *,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    coefficient_histories: dict[str, torch.Tensor],
    scene: str,
    forgetting_factor: float,
    initial_covariance: float,
    measurement_noise: float,
    start_index: int,
    n_example_points: int,
    include_fe_variants: bool = True,
    kalman_process_noise: float = 0.0,
    fe_sgd_learning_rate: float = 1.0,
    fe_sgd_momentum: float = 0.0,
    fe_sgd_weight_decay: float = 0.0,
    fe_window_size: int = 100,
    fe_window_ridge: float = 1e-6,
    linear_include_bias: bool = True,
) -> dict[str, object]:
    """Compute scalar metrics for all baseline predictions."""

    target_delta = target.squeeze(0)
    zero_prediction = predictions.get("zero_delta", torch.zeros_like(target))
    zero_error = torch.linalg.norm(zero_prediction.squeeze(0) - target_delta, dim=-1)

    method_summaries: dict[str, object] = {}
    for name, prediction in predictions.items():
        prediction_delta = prediction.squeeze(0)
        error = torch.linalg.norm(prediction_delta - target_delta, dim=-1)
        method_summaries[name] = {
            "label": METHOD_LABELS.get(name, name),
            "mean_error": float(error.mean()),
            "final_accumulated_error": float(error.sum()),
            "mean_error_to_zero_delta_ratio": _safe_ratio(
                float(error.mean()),
                float(zero_error.mean()),
            ),
            "mse": float(torch.nn.functional.mse_loss(prediction_delta, target_delta)),
        }

    coefficient_summaries: dict[str, object] = {}
    for name, history in coefficient_histories.items():
        norms = torch.linalg.norm(history, dim=-1)
        coefficient_summaries[name] = {
            "label": METHOD_LABELS.get(name, name),
            "n_coeff": int(history.shape[-1]),
            "final_norm": float(norms[-1]),
            "mean_norm": float(norms.mean()),
            "max_norm": float(norms.max()),
            "final": [float(value) for value in history[-1]],
        }

    return {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "start_index": start_index,
        "n_example_points": n_example_points,
        "methods": method_summaries,
        "coefficients": coefficient_summaries,
        "rls_parameters": {
            "forgetting_factor": forgetting_factor,
            "initial_covariance": initial_covariance,
            "measurement_noise": measurement_noise,
        },
        "fe_variant_parameters": {
            "include": include_fe_variants,
            "kalman_process_noise": kalman_process_noise,
            "sgd_learning_rate": fe_sgd_learning_rate,
            "sgd_momentum": fe_sgd_momentum,
            "sgd_weight_decay": fe_sgd_weight_decay,
            "window_size": fe_window_size,
            "window_ridge": fe_window_ridge,
        },
        "linear_baseline": {
            "include_bias": linear_include_bias,
        },
    }


def write_online_baseline_artifacts(
    artifact_dir: Path,
    *,
    scene: str,
    time: torch.Tensor,
    dt: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    coefficient_histories: dict[str, torch.Tensor],
    summary: dict[str, object],
) -> None:
    """Write CSV, JSON, and plot artifacts for online baseline comparisons."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_streaming_predictions_csv(
        artifact_dir / "streaming_predictions.csv",
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_error_plot(
        artifact_dir / "streaming_error.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_components_plot(
        artifact_dir / "streaming_components.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_delta_scale_plot(
        artifact_dir / "streaming_delta_scale.png",
        scene=scene,
        time=time,
        target=target,
        predictions=predictions,
    )
    write_streaming_trajectory_plot(
        artifact_dir / "streaming_trajectory.png",
        scene=scene,
        target=target,
        predictions=predictions,
        include_offline=True,
    )
    write_streaming_trajectory_plot(
        artifact_dir / "streaming_trajectory_online.png",
        scene=scene,
        target=target,
        predictions=predictions,
        include_offline=False,
    )
    write_coefficient_norm_plot(
        artifact_dir / "coefficient_norms.png",
        scene=scene,
        time=time,
        coefficient_histories=coefficient_histories,
    )
    write_delta_time_summary(
        artifact_dir / "trajectory_summary.json",
        scene=scene,
        dt=dt,
        target=target,
        predictions=predictions,
    )


def write_streaming_predictions_csv(
    path: Path,
    *,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write target and prediction values for each streaming step."""

    target_delta = target.squeeze(0)
    prediction_delta = {name: value.squeeze(0) for name, value in predictions.items()}
    errors = {
        name: torch.linalg.norm(value - target_delta, dim=-1)
        for name, value in prediction_delta.items()
    }

    with path.open("w", newline="") as f:
        fieldnames = ["index", "time"]
        fieldnames += [f"target_{idx}" for idx in range(target_delta.shape[-1])]
        for name in predictions:
            fieldnames.append(f"{name}_error")
            fieldnames += [f"{name}_prediction_{idx}" for idx in range(target_delta.shape[-1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(target_delta.shape[0]):
            row = {
                "index": idx,
                "time": float(time[idx]),
            }
            row.update(
                {
                    f"target_{dim}": float(target_delta[idx, dim])
                    for dim in range(target_delta.shape[-1])
                }
            )
            for name, prediction in prediction_delta.items():
                row[f"{name}_error"] = float(errors[name][idx])
                row.update(
                    {
                        f"{name}_prediction_{dim}": float(prediction[idx, dim])
                        for dim in range(target_delta.shape[-1])
                    }
                )
            writer.writerow(row)


def write_streaming_error_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write per-step and accumulated error for every compared method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, prediction in predictions.items():
        error = torch.linalg.norm(prediction.squeeze(0) - target_delta, dim=-1)
        kwargs = _plot_kwargs(name)
        axes[0].plot(time, error, label=METHOD_LABELS.get(name, name), **kwargs)
        axes[1].plot(time, torch.cumsum(error, dim=0), label=METHOD_LABELS.get(name, name), **kwargs)
    axes[0].set_ylabel("error norm")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("accumulated error")
    axes[1].legend(ncol=2)
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
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write per-dimension traces for the main online baselines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    selected = _selected_prediction_names(predictions)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for dim, ax in enumerate(axes.ravel()):
        ax.plot(time, target_delta[:, dim], label="target", linewidth=1.3, color="black")
        for name in selected:
            ax.plot(
                time,
                predictions[name].squeeze(0)[:, dim],
                label=METHOD_LABELS.get(name, name),
                **_plot_kwargs(name),
            )
        ax.set_ylabel(f"dim {dim}")
    axes.ravel()[0].legend(ncol=2)
    axes.ravel()[-1].set_xlabel("relative time [s]")
    axes.ravel()[-2].set_xlabel("relative time [s]")
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_delta_scale_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write delta norm and cumulative planar distance for compared methods."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_delta = target.squeeze(0)
    selected = _selected_prediction_names(predictions, include_offline=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, torch.linalg.norm(target_delta, dim=-1), label="target", color="black")
    axes[1].plot(
        time,
        torch.cumsum(torch.linalg.norm(target_delta[:, :2], dim=-1), dim=0),
        label="target",
        color="black",
    )
    for name in selected:
        prediction = predictions[name].squeeze(0)
        axes[0].plot(
            time,
            torch.linalg.norm(prediction, dim=-1),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
        axes[1].plot(
            time,
            torch.cumsum(torch.linalg.norm(prediction[:, :2], dim=-1), dim=0),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    axes[0].set_ylabel("delta norm")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("relative time [s]")
    axes[1].set_ylabel("cumulative planar delta")
    axes[1].legend(ncol=2)
    fig.suptitle(scene)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_streaming_trajectory_plot(
    path: Path,
    *,
    scene: str,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    include_offline: bool,
) -> None:
    """Write integrated local trajectory comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_pose = integrate_planar_deltas(target.squeeze(0))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(target_pose[:, 0], target_pose[:, 1], label="target", linewidth=1.6, color="black")
    for name in _selected_prediction_names(predictions, include_offline=include_offline):
        pose = integrate_planar_deltas(predictions[name].squeeze(0))
        ax.plot(
            pose[:, 0],
            pose[:, 1],
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    ax.scatter(target_pose[0, 0], target_pose[0, 1], s=20, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("integrated local x")
    ax.set_ylabel("integrated local y")
    title = scene if include_offline else f"{scene}: online methods"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_coefficient_norm_plot(
    path: Path,
    *,
    scene: str,
    time: torch.Tensor,
    coefficient_histories: dict[str, torch.Tensor],
) -> None:
    """Write coefficient norm histories for online methods."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, history in coefficient_histories.items():
        ax.plot(
            time,
            torch.linalg.norm(history, dim=-1),
            label=METHOD_LABELS.get(name, name),
            **_plot_kwargs(name),
        )
    ax.set_xlabel("relative time [s]")
    ax.set_ylabel("coefficient norm")
    ax.set_title(scene)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_delta_time_summary(
    path: Path,
    *,
    scene: str,
    dt: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    """Write compact trajectory-scale metrics for every method."""

    target_delta = target.squeeze(0)
    payload: dict[str, object] = {
        "scene": scene,
        "n_steps": int(target_delta.shape[0]),
        "dt": {
            "mean": float(dt.mean()),
            "min": float(dt.min()),
            "max": float(dt.max()),
            "total": float(dt.sum()),
        },
        "target_path_length": _path_length(target_delta),
        "methods": {},
    }
    for name, prediction in predictions.items():
        prediction_delta = prediction.squeeze(0)
        payload["methods"][name] = {
            "label": METHOD_LABELS.get(name, name),
            "prediction_path_length": _path_length(prediction_delta),
            "prediction_to_target_path_length_ratio": _safe_ratio(
                _path_length(prediction_delta),
                _path_length(target_delta),
            ),
            "prediction_delta_norm_mean": float(
                torch.linalg.norm(prediction_delta, dim=-1).mean()
            ),
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _selected_prediction_names(
    predictions: dict[str, torch.Tensor],
    *,
    include_offline: bool = False,
) -> list[str]:
    order = [
        "fe_rls",
        "fe_kalman",
        "fe_window_ls",
        "neuralfly_rls",
        "linear_rls",
        "fe_sgd",
        "static_node",
    ]
    if include_offline:
        order += ["offline_fe", "offline_neuralfly", "offline_linear"]
    return [name for name in order if name in predictions]


def _plot_kwargs(name: str) -> dict[str, object]:
    if name == "zero_delta":
        return {"linewidth": 1.0, "alpha": 0.45, "linestyle": "--"}
    if name.startswith("offline"):
        return {"linewidth": 1.0, "alpha": 0.7, "linestyle": ":"}
    if name == "static_node":
        return {"linewidth": 1.1, "alpha": 0.8, "linestyle": "-."}
    if name in {"fe_sgd", "fe_window_ls"}:
        return {"linewidth": 1.0, "alpha": 0.85}
    return {"linewidth": 1.2}


def _path_length(deltas: torch.Tensor) -> float:
    poses = integrate_planar_deltas(deltas)
    return float(torch.linalg.norm(poses[1:, :2] - poses[:-1, :2], dim=-1).sum())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
