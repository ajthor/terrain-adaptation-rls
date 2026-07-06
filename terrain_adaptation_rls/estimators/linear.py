"""Model-agnostic estimators for linear-in-coefficients predictions.

These estimators assume observations of the form

    y = Phi theta + noise

where ``theta`` is a low-dimensional online state. The feature provider is
model-specific: FE basis functions, NeuralFly-style features, and hand-designed
linear bases can all feed the same estimators as long as they expose ``Phi``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RLSState:
    """State for recursive least-squares coefficient adaptation."""

    coefficients: torch.Tensor
    covariance: torch.Tensor


@dataclass
class KalmanState:
    """State for Kalman-style coefficient adaptation."""

    coefficients: torch.Tensor
    covariance: torch.Tensor


@dataclass
class CoefficientSGDState:
    """State for gradient updates on a linear coefficient vector."""

    coefficients: torch.Tensor
    momentum: Optional[torch.Tensor] = None


@dataclass
class WindowedLeastSquaresState:
    """State for sliding-window ridge least squares."""

    coefficients: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor


def linear_predict(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Predict with a linear-in-coefficients model.

    Args:
        features: Basis/features with shape ``[..., output_dim, n_coeff]``.
        coefficients: Coefficients with shape ``[batch, n_coeff]``.

    Returns:
        Predictions with shape ``[..., output_dim]``.
    """

    if features.shape[-1] != coefficients.shape[-1]:
        raise ValueError(
            "Feature coefficient dimension does not match coefficient vector: "
            f"{features.shape[-1]} != {coefficients.shape[-1]}"
        )
    if features.shape[0] != coefficients.shape[0]:
        raise ValueError(
            "Feature batch dimension does not match coefficient batch dimension: "
            f"{features.shape[0]} != {coefficients.shape[0]}"
        )

    # features: [B, ..., D, K], coefficients: [B, K] -> [B, ..., D]
    return torch.einsum("b...dk,bk->b...d", features, coefficients)


def rls_update(
    state: RLSState,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    forgetting_factor: float = 1.0,
    measurement_noise: float | torch.Tensor = 1e-6,
) -> RLSState:
    """Apply one multi-output RLS update.

    Args:
        state: Current coefficients and covariance.
        features: Observation matrix with shape ``[batch, output_dim, n_coeff]``.
        target: Observation target with shape ``[batch, output_dim]``.
        forgetting_factor: Exponential forgetting factor in ``(0, 1]``.
        measurement_noise: Scalar or batched observation covariance.
    """

    if forgetting_factor <= 0.0 or forgetting_factor > 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")

    covariance = state.covariance / forgetting_factor
    return _linear_bayes_update(
        coefficients=state.coefficients,
        covariance=covariance,
        features=features,
        target=target,
        measurement_noise=measurement_noise,
        state_type=RLSState,
    )


def kalman_update(
    state: KalmanState,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    process_noise: float | torch.Tensor = 0.0,
    measurement_noise: float | torch.Tensor = 1e-6,
) -> KalmanState:
    """Apply one Kalman-style linear coefficient update."""

    covariance = state.covariance + _covariance_like(
        state.covariance,
        process_noise,
        state.covariance.shape[-1],
    )
    return _linear_bayes_update(
        coefficients=state.coefficients,
        covariance=covariance,
        features=features,
        target=target,
        measurement_noise=measurement_noise,
        state_type=KalmanState,
    )


def coefficient_sgd_update(
    state: CoefficientSGDState,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> CoefficientSGDState:
    """Apply one gradient step on the coefficient vector."""

    prediction = linear_predict(features, state.coefficients)
    residual = prediction - target
    grad = torch.einsum("bdk,bd->bk", features, residual) / target.shape[-1]
    if weight_decay:
        grad = grad + weight_decay * state.coefficients

    velocity = state.momentum
    if momentum:
        if velocity is None:
            velocity = torch.zeros_like(state.coefficients)
        velocity = momentum * velocity + grad
        grad_step = velocity
    else:
        grad_step = grad
        velocity = None

    return CoefficientSGDState(
        coefficients=state.coefficients - learning_rate * grad_step,
        momentum=velocity,
    )


def windowed_least_squares_update(
    state: WindowedLeastSquaresState,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    ridge: float = 1e-6,
) -> WindowedLeastSquaresState:
    """Append an observation and solve ridge least squares over a window."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    feature_window = torch.cat([state.features, features.unsqueeze(1)], dim=1)
    target_window = torch.cat([state.targets, target.unsqueeze(1)], dim=1)
    feature_window = feature_window[:, -window_size:]
    target_window = target_window[:, -window_size:]

    coefficients = solve_ridge_coefficients(feature_window, target_window, ridge=ridge)
    return WindowedLeastSquaresState(
        coefficients=coefficients,
        features=feature_window,
        targets=target_window,
    )


def solve_ridge_coefficients(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Solve batched ridge least squares for windowed observations.

    Args:
        features: Shape ``[batch, window, output_dim, n_coeff]``.
        targets: Shape ``[batch, window, output_dim]``.
        ridge: Non-negative ridge penalty.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if features.shape[:-1] != targets.shape:
        raise ValueError(
            "features and targets must share batch/window/output dimensions: "
            f"{features.shape[:-1]} != {targets.shape}"
        )

    batch_size = features.shape[0]
    n_coeff = features.shape[-1]
    design = features.reshape(batch_size, -1, n_coeff)
    response = targets.reshape(batch_size, -1)

    gram = torch.matmul(design.transpose(-1, -2), design)
    rhs = torch.matmul(design.transpose(-1, -2), response.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(n_coeff, dtype=features.dtype, device=features.device)
    gram = gram + ridge * eye.unsqueeze(0)
    return torch.linalg.solve(gram, rhs)


def _linear_bayes_update(
    *,
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
    measurement_noise: float | torch.Tensor,
    state_type: type[RLSState] | type[KalmanState],
) -> RLSState | KalmanState:
    _check_update_shapes(coefficients, covariance, features, target)

    output_dim = target.shape[-1]
    noise = _covariance_like(covariance, measurement_noise, output_dim)
    innovation = target - linear_predict(features, coefficients)
    innovation_cov = torch.matmul(
        torch.matmul(features, covariance),
        features.transpose(-1, -2),
    ) + noise

    gain_t = torch.linalg.solve(
        innovation_cov,
        torch.matmul(features, covariance),
    )
    gain = gain_t.transpose(-1, -2)

    new_coefficients = coefficients + torch.matmul(
        gain,
        innovation.unsqueeze(-1),
    ).squeeze(-1)
    new_covariance = covariance - torch.matmul(
        torch.matmul(gain, features),
        covariance,
    )
    new_covariance = 0.5 * (new_covariance + new_covariance.transpose(-1, -2))
    return state_type(coefficients=new_coefficients, covariance=new_covariance)


def _check_update_shapes(
    coefficients: torch.Tensor,
    covariance: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [batch, output_dim, n_coeff], got {features.shape}")
    if target.ndim != 2:
        raise ValueError(f"target must have shape [batch, output_dim], got {target.shape}")
    if coefficients.ndim != 2:
        raise ValueError(f"coefficients must have shape [batch, n_coeff], got {coefficients.shape}")
    if covariance.ndim != 3:
        raise ValueError(f"covariance must have shape [batch, n_coeff, n_coeff], got {covariance.shape}")
    if features.shape[:2] != target.shape:
        raise ValueError(f"features and target output dimensions differ: {features.shape[:2]} != {target.shape}")
    if features.shape[0] != coefficients.shape[0] or features.shape[0] != covariance.shape[0]:
        raise ValueError("batch dimensions of features, coefficients, and covariance must match")
    if features.shape[-1] != coefficients.shape[-1]:
        raise ValueError("feature and coefficient dimensions must match")
    if covariance.shape[-2:] != (coefficients.shape[-1], coefficients.shape[-1]):
        raise ValueError("covariance shape must match coefficient dimension")


def _covariance_like(
    reference: torch.Tensor,
    value: float | torch.Tensor,
    dim: int,
) -> torch.Tensor:
    batch_size = reference.shape[0]
    if isinstance(value, torch.Tensor):
        value = value.to(dtype=reference.dtype, device=reference.device)
        if value.ndim == 0:
            return value * _batch_eye(batch_size, dim, reference)
        if value.ndim == 2:
            return value.unsqueeze(0).expand(batch_size, dim, dim)
        if value.ndim == 3:
            return value
        raise ValueError(f"Unsupported covariance tensor shape: {value.shape}")
    return float(value) * _batch_eye(batch_size, dim, reference)


def _batch_eye(batch_size: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(dim, dtype=reference.dtype, device=reference.device)
    return eye.unsqueeze(0).expand(batch_size, dim, dim)
