"""Prediction metrics shared by streaming and baseline evaluations."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from terrain_adaptation_rls.evaluation.diagnostic_plots import integrate_planar_deltas


STATE_LABELS = ("x", "y", "yaw", "x_velocity", "y_velocity", "yaw_rate")
DEFAULT_LOGGED_K_STEP_HORIZONS = (1, 5, 10, 20, 50)
DEFAULT_ADAPTATION_WINDOW = 10
DEFAULT_ADAPTATION_THRESHOLDS = (0.25, 0.50)


def summarize_prediction_metrics(
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dt: torch.Tensor | None = None,
    zero_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute scalar one-step and integrated-trajectory metrics."""

    target_delta = _single_trajectory(target)
    prediction_delta = _single_trajectory(prediction)
    if target_delta.shape != prediction_delta.shape:
        raise ValueError(
            "target and prediction shapes must match: "
            f"{tuple(target_delta.shape)} != {tuple(prediction_delta.shape)}"
        )

    error_delta = prediction_delta - target_delta
    error_norm = torch.linalg.norm(error_delta, dim=-1)
    planar_error_norm = torch.linalg.norm(error_delta[:, :2], dim=-1)
    velocity_error_norm = torch.linalg.norm(error_delta[:, 3:6], dim=-1)
    target_pose = integrate_planar_deltas(target_delta)
    prediction_pose = integrate_planar_deltas(prediction_delta)
    pose_error = prediction_pose - target_pose
    position_error = torch.linalg.norm(pose_error[:, :2], dim=-1)
    yaw_error = torch.abs(pose_error[:, 2])
    target_path_length = _path_length(target_pose)
    prediction_path_length = _path_length(prediction_pose)
    component_abs_error = torch.abs(error_delta)
    component_bias = error_delta.mean(dim=0)
    adaptation_metrics = summarize_adaptation_time_metrics(
        error_norm,
        dt=dt,
        window=DEFAULT_ADAPTATION_WINDOW,
        improvement_thresholds=DEFAULT_ADAPTATION_THRESHOLDS,
    )

    metrics: dict[str, float] = {
        "mean_error": float(error_norm.mean()),
        "median_error": float(error_norm.median()),
        "rmse_error_norm": float(torch.sqrt(torch.mean(error_norm.square()))),
        "p90_error": _quantile(error_norm, 0.90),
        "p95_error": _quantile(error_norm, 0.95),
        "max_error": float(error_norm.max()),
        "final_accumulated_error": float(error_norm.sum()),
        "mse": float(torch.nn.functional.mse_loss(prediction_delta, target_delta)),
        "mae": float(component_abs_error.mean()),
        "max_abs_component_error": float(component_abs_error.max()),
        "bias_norm": float(torch.linalg.norm(component_bias)),
        "planar_mean_error": float(planar_error_norm.mean()),
        "planar_rmse_error": float(torch.sqrt(torch.mean(planar_error_norm.square()))),
        "velocity_mean_error": float(velocity_error_norm.mean()),
        "velocity_rmse_error": float(torch.sqrt(torch.mean(velocity_error_norm.square()))),
        "yaw_mae": float(torch.abs(error_delta[:, 2]).mean()),
        "yaw_rate_mae": float(torch.abs(error_delta[:, 5]).mean()),
        "integrated_position_mean_error": float(position_error.mean()),
        "integrated_position_rmse_error": float(
            torch.sqrt(torch.mean(position_error.square()))
        ),
        "integrated_position_max_error": float(position_error.max()),
        "integrated_position_final_error": float(position_error[-1]),
        "integrated_yaw_mean_abs_error": float(yaw_error.mean()),
        "integrated_yaw_final_abs_error": float(yaw_error[-1]),
        "target_path_length": target_path_length,
        "prediction_path_length": prediction_path_length,
        "path_length_abs_error": abs(prediction_path_length - target_path_length),
        "prediction_to_target_path_length_ratio": _safe_ratio(
            prediction_path_length,
            target_path_length,
        ),
        "endpoint_position_error": float(position_error[-1]),
        **adaptation_metrics,
    }
    if dt is not None:
        dt_single = _single_time(dt)
        metrics.update(
            {
                "duration": float(dt_single.sum()),
                "mean_dt": float(dt_single.mean()),
            }
        )

    metrics.update(_per_dimension_metrics(error_delta))
    if zero_metrics is not None:
        metrics.update(_zero_ratios(metrics, zero_metrics))
    return metrics


def summarize_adaptation_time_metrics(
    errors: torch.Tensor,
    *,
    dt: torch.Tensor | None = None,
    window: int = DEFAULT_ADAPTATION_WINDOW,
    improvement_thresholds: Iterable[float] = DEFAULT_ADAPTATION_THRESHOLDS,
) -> dict[str, float]:
    """Summarize how quickly an online method improves from its initial error.

    The crossing time is the first sample where the moving-average error is at
    least ``threshold`` better than the first-window moving-average error. If a
    method never crosses the threshold, the sample count is ``n_steps + 1`` and
    the corresponding ``reached`` field is 0.
    """

    error = errors.flatten().float()
    n_steps = int(error.numel())
    if n_steps == 0:
        return {
            "adaptation_window": float(max(1, int(window))),
            "adaptation_initial_error": 0.0,
            "adaptation_final_error": 0.0,
            "adaptation_final_improvement_fraction": 0.0,
            "adaptation_mean_improvement_fraction": 0.0,
        }

    window = max(1, min(int(window), n_steps))
    smoothed = _moving_average(error, window)
    reference = torch.clamp(smoothed[0], min=1e-12)
    improvement = (reference - smoothed) / reference
    metrics = {
        "adaptation_window": float(window),
        "adaptation_initial_error": float(reference),
        "adaptation_final_error": float(smoothed[-1]),
        "adaptation_final_improvement_fraction": float(improvement[-1]),
        "adaptation_mean_improvement_fraction": float(improvement.mean()),
    }

    dt_single = _single_time(dt) if dt is not None else None
    if dt_single is not None and dt_single.numel() != n_steps:
        raise ValueError(
            "dt and error length must match: "
            f"{dt_single.numel()} != {n_steps}"
        )

    for threshold in improvement_thresholds:
        threshold = float(threshold)
        key = _threshold_key(threshold)
        crossing = torch.nonzero(improvement >= threshold, as_tuple=False)
        if crossing.numel() == 0:
            sample_count = n_steps + 1
            reached = 0.0
            seconds = (
                float(dt_single.sum() + dt_single.mean())
                if dt_single is not None
                else float(sample_count)
            )
        else:
            crossing_index = int(crossing[0, 0])
            sample_count = crossing_index + window
            reached = 1.0
            seconds = (
                float(dt_single[:sample_count].sum())
                if dt_single is not None
                else float(sample_count)
            )
        metrics[f"adaptation_samples_to_{key}_improvement"] = float(sample_count)
        metrics[f"adaptation_seconds_to_{key}_improvement"] = float(seconds)
        metrics[f"adaptation_reached_{key}_improvement"] = reached
    return metrics


def summarize_logged_k_step_metrics(
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dt: torch.Tensor | None = None,
    horizons: Iterable[int] = DEFAULT_LOGGED_K_STEP_HORIZONS,
) -> dict[str, float]:
    """Summarize cumulative logged-input lookahead errors for several horizons."""

    target_delta = _single_trajectory(target)
    prediction_delta = _single_trajectory(prediction)
    if target_delta.shape != prediction_delta.shape:
        raise ValueError(
            "target and prediction shapes must match: "
            f"{tuple(target_delta.shape)} != {tuple(prediction_delta.shape)}"
        )

    dt_single = _single_time(dt) if dt is not None else None
    if dt_single is not None and dt_single.shape[0] != target_delta.shape[0]:
        raise ValueError(
            "dt and trajectory length must match: "
            f"{dt_single.shape[0]} != {target_delta.shape[0]}"
        )

    metrics: dict[str, float] = {}
    n_steps = target_delta.shape[0]
    for horizon in _valid_horizons(horizons, n_steps):
        endpoint_errors: list[float] = []
        accumulated_errors: list[float] = []
        trajectory_rmses: list[float] = []
        integral_square_errors: list[float] = []
        final_position_errors: list[float] = []
        final_yaw_errors: list[float] = []

        for start in range(0, n_steps - horizon + 1):
            target_window = target_delta[start : start + horizon]
            prediction_window = prediction_delta[start : start + horizon]
            target_cumulative = torch.cumsum(target_window, dim=0)
            prediction_cumulative = torch.cumsum(prediction_window, dim=0)
            cumulative_error = torch.linalg.norm(
                prediction_cumulative - target_cumulative,
                dim=-1,
            )
            weights = (
                dt_single[start : start + horizon]
                if dt_single is not None
                else torch.ones_like(cumulative_error)
            )
            square_error = cumulative_error.square()
            square_integral = torch.sum(square_error * weights)
            endpoint_errors.append(float(cumulative_error[-1]))
            accumulated_errors.append(float(cumulative_error.sum()))
            trajectory_rmses.append(
                float(torch.sqrt(square_integral / torch.clamp(weights.sum(), min=1e-12)))
            )
            integral_square_errors.append(float(square_integral))

            target_pose = integrate_planar_deltas(target_window)
            prediction_pose = integrate_planar_deltas(prediction_window)
            final_pose_error = prediction_pose[-1] - target_pose[-1]
            final_position_errors.append(float(torch.linalg.norm(final_pose_error[:2])))
            final_yaw_errors.append(float(torch.abs(final_pose_error[2])))

        prefix = f"logged_k{horizon}"
        metrics[f"{prefix}_n_windows"] = float(len(endpoint_errors))
        metrics[f"{prefix}_endpoint_error_mean"] = _mean(endpoint_errors)
        metrics[f"{prefix}_endpoint_error_median"] = _median(endpoint_errors)
        metrics[f"{prefix}_endpoint_error_p95"] = _quantile_list(endpoint_errors, 0.95)
        metrics[f"{prefix}_accumulated_error_mean"] = _mean(accumulated_errors)
        metrics[f"{prefix}_accumulated_error_median"] = _median(accumulated_errors)
        metrics[f"{prefix}_accumulated_error_p95"] = _quantile_list(
            accumulated_errors,
            0.95,
        )
        metrics[f"{prefix}_trajectory_rmse_mean"] = _mean(trajectory_rmses)
        metrics[f"{prefix}_trajectory_rmse_median"] = _median(trajectory_rmses)
        metrics[f"{prefix}_trajectory_rmse_p95"] = _quantile_list(
            trajectory_rmses,
            0.95,
        )
        metrics[f"{prefix}_integral_square_error_mean"] = _mean(integral_square_errors)
        metrics[f"{prefix}_integral_square_error_median"] = _median(
            integral_square_errors
        )
        metrics[f"{prefix}_integral_square_error_p95"] = _quantile_list(
            integral_square_errors,
            0.95,
        )
        metrics[f"{prefix}_final_position_error_mean"] = _mean(final_position_errors)
        metrics[f"{prefix}_final_yaw_error_mean"] = _mean(final_yaw_errors)
    return metrics


def _zero_ratios(
    metrics: dict[str, float],
    zero_metrics: dict[str, float],
) -> dict[str, float]:
    ratio_keys = {
        "mean_error": "mean_error_to_zero_delta_ratio",
        "final_accumulated_error": "accumulated_error_to_zero_delta_ratio",
        "mse": "mse_to_zero_delta_ratio",
        "integrated_position_mean_error": "integrated_position_mean_error_to_zero_delta_ratio",
        "integrated_position_final_error": "integrated_position_final_error_to_zero_delta_ratio",
        "endpoint_position_error": "endpoint_position_error_to_zero_delta_ratio",
    }
    return {
        output_key: _safe_ratio(metrics[source_key], zero_metrics[source_key])
        for source_key, output_key in ratio_keys.items()
        if source_key in metrics and source_key in zero_metrics
    }


def _per_dimension_metrics(error_delta: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n_dims = min(error_delta.shape[-1], len(STATE_LABELS))
    for index in range(n_dims):
        label = STATE_LABELS[index]
        values = error_delta[:, index]
        metrics[f"{label}_mae"] = float(torch.abs(values).mean())
        metrics[f"{label}_rmse"] = float(torch.sqrt(torch.mean(values.square())))
        metrics[f"{label}_bias"] = float(values.mean())
        metrics[f"{label}_max_abs_error"] = float(torch.abs(values).max())
    return metrics


def _single_trajectory(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {tensor.shape[0]}")
        return tensor.squeeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"expected trajectory shape [steps, dim] or [1, steps, dim], got {tuple(tensor.shape)}")


def _single_time(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {tensor.shape[0]}")
        return tensor.squeeze(0)
    if tensor.ndim == 1:
        return tensor
    raise ValueError(f"expected time shape [steps] or [1, steps], got {tuple(tensor.shape)}")


def _moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    if values.numel() < window:
        return values.mean().view(1)
    kernel = torch.ones(window, dtype=values.dtype, device=values.device) / window
    return torch.nn.functional.conv1d(
        values.view(1, 1, -1),
        kernel.view(1, 1, -1),
    ).view(-1)


def _threshold_key(threshold: float) -> str:
    return f"{int(round(100.0 * threshold))}pct"


def _valid_horizons(horizons: Iterable[int], n_steps: int) -> list[int]:
    return sorted({int(horizon) for horizon in horizons if 0 < int(horizon) <= n_steps})


def _path_length(poses: torch.Tensor) -> float:
    return float(torch.linalg.norm(poses[1:, :2] - poses[:-1, :2], dim=-1).sum())


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q))


def _quantile_list(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return _quantile(tensor, q)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.median())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
