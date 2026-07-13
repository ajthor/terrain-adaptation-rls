"""Runtime methods for Phoenix-shaped online dynamics adaptation.

This module is the single place to look for the runtime contract and the
coefficient-adaptation wrappers used by FE-RLS, NeuralFly-style RLS, and simple
linear baselines. The small registry in ``methods.registry`` is only for config
validation and reviewer-facing method names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal hosts.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RuntimeInput:
    """Inputs passed to a runtime dynamics model."""

    xs: object
    dt: object


@dataclass(frozen=True)
class Observation:
    """One logged observation for streaming evaluation."""

    inputs: RuntimeInput
    target: object
    time: float | None = None


class RuntimeMethod(Protocol):
    """Model-plus-update wrapper used by streaming and rollout eval."""

    def initial_state(self) -> object:
        """Return the initial online adaptation state."""

    def predict(self, state: object, inputs: RuntimeInput) -> object:
        """Predict from the current state before seeing the current target."""

    def update(self, state: object, observation: Observation) -> object:
        """Update online state after the current prediction is recorded."""


UpdateRule = Literal["rls", "kalman", "sgd", "window_ls"]


def _missing_torch() -> None:
    raise ModuleNotFoundError(
        "Runtime coefficient methods require torch. Use the devcontainer or install "
        "project dependencies before constructing runtime methods."
    )


if torch is None:
    CoefficientState = object

    class FeatureProvider(Protocol):
        """Callable feature provider for coefficient-adaptation methods."""

        n_coeff: int

        def __call__(self, inputs: RuntimeInput) -> object:
            """Return features shaped [batch, ..., output_dim, n_coeff]."""

    class LinearBasisProvider:  # pragma: no cover - import shim for no-torch hosts.
        def __init__(self, *args: object, **kwargs: object) -> None:
            _missing_torch()

    class NeuralFlyStyleBasisProvider:  # pragma: no cover
        def __init__(self, *args: object, **kwargs: object) -> None:
            _missing_torch()

    class ALPaCABasisProvider:  # pragma: no cover
        def __init__(self, *args: object, **kwargs: object) -> None:
            _missing_torch()

    class FunctionEncoderBasisProvider:  # pragma: no cover
        def __init__(self, *args: object, **kwargs: object) -> None:
            _missing_torch()

    class TorchCoefficientMethod:  # pragma: no cover
        def __init__(self, *args: object, **kwargs: object) -> None:
            _missing_torch()

    def concatenate_runtime_input(
        inputs: RuntimeInput,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> object:
        _missing_torch()

    def build_runtime_method(name: str, **kwargs: Any) -> object:
        _missing_torch()

else:
    from terrain_adaptation_rls.estimators.linear import (
        CoefficientSGDState,
        KalmanState,
        RLSState,
        WindowedLeastSquaresState,
        coefficient_sgd_update,
        kalman_update,
        linear_predict,
        rls_update,
        windowed_least_squares_update,
    )

    CoefficientState = RLSState | KalmanState | CoefficientSGDState | WindowedLeastSquaresState

    class FeatureProvider(Protocol):
        """Callable feature provider for coefficient-adaptation methods."""

        n_coeff: int

        def __call__(self, inputs: RuntimeInput) -> torch.Tensor:
            """Return features shaped [batch, ..., output_dim, n_coeff]."""

    def concatenate_runtime_input(
        inputs: RuntimeInput,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Concatenate Phoenix-style ``(xs, dt)`` inputs into one feature tensor."""

        xs = _as_tensor(inputs.xs, dtype=dtype, device=device)
        dt = _as_tensor(inputs.dt, dtype=xs.dtype, device=xs.device)

        if xs.ndim == 1:
            xs = xs.unsqueeze(0)
        if dt.ndim == 0:
            dt = dt.unsqueeze(0)

        if dt.ndim == xs.ndim - 1:
            dt = dt.unsqueeze(-1)
        elif dt.ndim == xs.ndim and dt.shape[-1] == 1:
            pass
        else:
            raise ValueError(
                "dt must have either one fewer dimension than xs or a trailing singleton "
                f"dimension; got xs={tuple(xs.shape)}, dt={tuple(dt.shape)}"
            )

        if xs.shape[:-1] != dt.shape[:-1]:
            raise ValueError(
                "xs and dt batch/point dimensions must match: "
                f"{tuple(xs.shape[:-1])} != {tuple(dt.shape[:-1])}"
            )

        return torch.cat([xs, dt], dim=-1)

    @dataclass(frozen=True)
    class LinearBasisProvider:
        """Block-diagonal linear basis over concatenated ``(xs, dt)`` input."""

        input_dim: int = 9
        output_dim: int = 6
        include_bias: bool = True

        @property
        def n_coeff(self) -> int:
            return self.output_dim * self.scalar_feature_dim

        @property
        def scalar_feature_dim(self) -> int:
            return self.input_dim + int(self.include_bias)

        def __call__(self, inputs: RuntimeInput) -> torch.Tensor:
            z = concatenate_runtime_input(inputs)
            if z.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected concatenated input dim {self.input_dim}, got {z.shape[-1]}"
                )

            if self.include_bias:
                bias = torch.ones(*z.shape[:-1], 1, dtype=z.dtype, device=z.device)
                z = torch.cat([z, bias], dim=-1)

            features = z.new_zeros(*z.shape[:-1], self.output_dim, self.n_coeff)
            for output_index in range(self.output_dim):
                start = output_index * self.scalar_feature_dim
                stop = start + self.scalar_feature_dim
                features[..., output_index, start:stop] = z
            return features

    class NeuralFlyStyleBasisProvider(torch.nn.Module):
        """Learned basis network with low-dimensional online coefficients."""

        def __init__(
            self,
            *,
            input_dim: int = 9,
            output_dim: int = 6,
            n_basis: int = 8,
            hidden_size: int = 128,
            n_hidden_layers: int = 2,
            activation_factory: Callable[[], torch.nn.Module] = torch.nn.ReLU,
        ) -> None:
            super().__init__()
            if n_basis <= 0:
                raise ValueError("n_basis must be positive")
            if n_hidden_layers < 0:
                raise ValueError("n_hidden_layers must be non-negative")

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_basis = n_basis

            layers: list[torch.nn.Module] = []
            width = input_dim
            for _ in range(n_hidden_layers):
                layers.append(torch.nn.Linear(width, hidden_size))
                layers.append(activation_factory())
                width = hidden_size
            layers.append(torch.nn.Linear(width, output_dim * n_basis))
            self.network = torch.nn.Sequential(*layers)

        @property
        def n_coeff(self) -> int:
            return self.n_basis

        def forward(self, inputs: RuntimeInput) -> torch.Tensor:
            z = concatenate_runtime_input(inputs)
            if z.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected concatenated input dim {self.input_dim}, got {z.shape[-1]}"
                )

            flat_z = z.reshape(-1, self.input_dim)
            flat_features = self.network(flat_z)
            return flat_features.reshape(*z.shape[:-1], self.output_dim, self.n_basis)

    class ALPaCABasisProvider(torch.nn.Module):
        """ALPaCA feature map and learned Gaussian prior over coefficients."""

        def __init__(
            self,
            *,
            input_dim: int = 9,
            output_dim: int = 6,
            n_basis: int = 8,
            hidden_size: int = 128,
            n_hidden_layers: int = 2,
            activation_factory: Callable[[], torch.nn.Module] = torch.nn.ReLU,
            initial_prior_variance: float = 1.0,
            initial_noise_variance: float = 1e-4,
        ) -> None:
            super().__init__()
            if n_basis <= 0:
                raise ValueError("n_basis must be positive")
            if n_hidden_layers < 0:
                raise ValueError("n_hidden_layers must be non-negative")
            if initial_prior_variance <= 0.0:
                raise ValueError("initial_prior_variance must be positive")
            if initial_noise_variance <= 0.0:
                raise ValueError("initial_noise_variance must be positive")

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.n_basis = n_basis
            self.prior_mean = torch.nn.Parameter(torch.zeros(n_basis))
            self.log_prior_variance = torch.nn.Parameter(
                torch.full((n_basis,), float(torch.log(torch.tensor(initial_prior_variance))))
            )
            self.log_noise_variance = torch.nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(initial_noise_variance))))
            )

            layers: list[torch.nn.Module] = []
            width = input_dim
            for _ in range(n_hidden_layers):
                layers.append(torch.nn.Linear(width, hidden_size))
                layers.append(activation_factory())
                width = hidden_size
            layers.append(torch.nn.Linear(width, output_dim * n_basis))
            self.network = torch.nn.Sequential(*layers)

        @property
        def n_coeff(self) -> int:
            return self.n_basis

        def forward(self, inputs: RuntimeInput) -> torch.Tensor:
            z = concatenate_runtime_input(inputs)
            if z.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected concatenated input dim {self.input_dim}, got {z.shape[-1]}"
                )

            flat_z = z.reshape(-1, self.input_dim)
            flat_features = self.network(flat_z)
            return flat_features.reshape(*z.shape[:-1], self.output_dim, self.n_basis)

        def prior_variance(self) -> torch.Tensor:
            return torch.exp(self.log_prior_variance).clamp_min(1e-10)

        def noise_variance(self) -> torch.Tensor:
            return torch.exp(self.log_noise_variance).clamp_min(1e-10)

    @dataclass(frozen=True)
    class FunctionEncoderBasisProvider:
        """Feature provider for upstream ``FunctionEncoder`` basis functions."""

        model: torch.nn.Module

        @property
        def n_coeff(self) -> int:
            basis_functions = getattr(self.model.basis_functions, "basis_functions", None)
            if basis_functions is None:
                raise AttributeError(
                    "FunctionEncoder model does not expose basis_functions.basis_functions"
                )
            return len(basis_functions)

        def __call__(self, inputs: RuntimeInput) -> torch.Tensor:
            return self.model.basis_functions((inputs.xs, inputs.dt))

    class TorchCoefficientMethod:
        """Runtime wrapper for torch basis providers and coefficient updates."""

        def __init__(
            self,
            feature_provider: FeatureProvider,
            *,
            update_rule: UpdateRule = "rls",
            output_dim: int | None = None,
            initial_covariance: float = 1_000.0,
            forgetting_factor: float = 0.95,
            measurement_noise: float = 1e-6,
            process_noise: float = 0.0,
            learning_rate: float = 1e-2,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            window_size: int = 100,
            ridge: float = 1e-6,
            initial_coefficients: torch.Tensor | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype = torch.float32,
        ) -> None:
            self.feature_provider = feature_provider
            self.update_rule = update_rule
            self.output_dim = output_dim
            self.initial_covariance = initial_covariance
            self.forgetting_factor = forgetting_factor
            self.measurement_noise = measurement_noise
            self.process_noise = process_noise
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.window_size = window_size
            self.ridge = ridge
            self.initial_coefficients = initial_coefficients
            self.device = torch.device(device) if device is not None else self._provider_device()
            self.dtype = dtype

            if update_rule not in {"rls", "kalman", "sgd", "window_ls"}:
                raise ValueError(f"Unknown update_rule '{update_rule}'")
            if update_rule == "window_ls" and output_dim is None:
                raise ValueError("output_dim is required for window_ls initial state")

        @property
        def n_coeff(self) -> int:
            return int(self.feature_provider.n_coeff)

        def initial_state(self) -> CoefficientState:
            """Return a single-batch initial coefficient state."""

            return self._new_state(batch_size=1)

        @torch.no_grad()
        def predict(self, state: CoefficientState, inputs: RuntimeInput) -> torch.Tensor:
            features = self.feature_provider(inputs).detach()
            state = self._match_state_batch(state, features.shape[0])
            return linear_predict(features, _state_coefficients(state))

        @torch.no_grad()
        def update(self, state: CoefficientState, observation: Observation) -> CoefficientState:
            features = self.feature_provider(observation.inputs).detach()
            target = _as_tensor(
                observation.target,
                dtype=features.dtype,
                device=features.device,
            ).detach()
            if target.ndim == 1:
                target = target.unsqueeze(0)

            state = self._match_state_batch(state, features.shape[0])
            for feature_step, target_step in _iter_point_observations(features, target):
                state = self._update_one(state, feature_step, target_step)
            return state

        def _update_one(
            self,
            state: CoefficientState,
            features: torch.Tensor,
            target: torch.Tensor,
        ) -> CoefficientState:
            if self.update_rule == "rls":
                if not isinstance(state, RLSState):
                    raise TypeError(f"Expected RLSState, got {type(state).__name__}")
                return rls_update(
                    state,
                    features,
                    target,
                    forgetting_factor=self.forgetting_factor,
                    measurement_noise=self.measurement_noise,
                )
            if self.update_rule == "kalman":
                if not isinstance(state, KalmanState):
                    raise TypeError(f"Expected KalmanState, got {type(state).__name__}")
                return kalman_update(
                    state,
                    features,
                    target,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            if self.update_rule == "sgd":
                if not isinstance(state, CoefficientSGDState):
                    raise TypeError(
                        f"Expected CoefficientSGDState, got {type(state).__name__}"
                    )
                return coefficient_sgd_update(
                    state,
                    features,
                    target,
                    learning_rate=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            if not isinstance(state, WindowedLeastSquaresState):
                raise TypeError(
                    f"Expected WindowedLeastSquaresState, got {type(state).__name__}"
                )
            return windowed_least_squares_update(
                state,
                features,
                target,
                window_size=self.window_size,
                ridge=self.ridge,
            )

        def _new_state(self, *, batch_size: int) -> CoefficientState:
            coefficients = self._initial_coefficients(batch_size)
            if self.update_rule == "rls":
                return RLSState(
                    coefficients=coefficients,
                    covariance=self.initial_covariance
                    * _batch_eye(batch_size, self.n_coeff, coefficients),
                )
            if self.update_rule == "kalman":
                return KalmanState(
                    coefficients=coefficients,
                    covariance=self.initial_covariance
                    * _batch_eye(batch_size, self.n_coeff, coefficients),
                )
            if self.update_rule == "sgd":
                return CoefficientSGDState(coefficients=coefficients)
            output_dim = int(self.output_dim)
            return WindowedLeastSquaresState(
                coefficients=coefficients,
                features=coefficients.new_empty(batch_size, 0, output_dim, self.n_coeff),
                targets=coefficients.new_empty(batch_size, 0, output_dim),
            )

        def _initial_coefficients(self, batch_size: int) -> torch.Tensor:
            if self.initial_coefficients is None:
                return torch.zeros(
                    batch_size,
                    self.n_coeff,
                    dtype=self.dtype,
                    device=self.device,
                )

            coefficients = self.initial_coefficients.to(dtype=self.dtype, device=self.device)
            if coefficients.ndim == 1:
                coefficients = coefficients.unsqueeze(0)
            if coefficients.shape[-1] != self.n_coeff:
                raise ValueError(
                    "initial_coefficients coefficient dimension does not match provider: "
                    f"{coefficients.shape[-1]} != {self.n_coeff}"
                )
            return _expand_batch(coefficients, batch_size)

        def _match_state_batch(
            self,
            state: CoefficientState,
            batch_size: int,
        ) -> CoefficientState:
            current_batch = _state_coefficients(state).shape[0]
            if current_batch == batch_size:
                return state
            if current_batch != 1:
                raise ValueError(
                    f"Cannot use state batch {current_batch} with input batch {batch_size}"
                )

            if isinstance(state, RLSState):
                return RLSState(
                    coefficients=_expand_batch(state.coefficients, batch_size),
                    covariance=_expand_batch(state.covariance, batch_size),
                )
            if isinstance(state, KalmanState):
                return KalmanState(
                    coefficients=_expand_batch(state.coefficients, batch_size),
                    covariance=_expand_batch(state.covariance, batch_size),
                )
            if isinstance(state, CoefficientSGDState):
                momentum = None if state.momentum is None else _expand_batch(
                    state.momentum,
                    batch_size,
                )
                return CoefficientSGDState(
                    coefficients=_expand_batch(state.coefficients, batch_size),
                    momentum=momentum,
                )
            return WindowedLeastSquaresState(
                coefficients=_expand_batch(state.coefficients, batch_size),
                features=_expand_batch(state.features, batch_size),
                targets=_expand_batch(state.targets, batch_size),
            )

        def _provider_device(self) -> torch.device:
            if isinstance(self.feature_provider, torch.nn.Module):
                try:
                    return next(self.feature_provider.parameters()).device
                except StopIteration:
                    pass
            return torch.device("cpu")

    def build_runtime_method(name: str, **kwargs: Any) -> object:
        """Build a runtime method by reviewer-facing method name."""

        if name == "linear_basis_rls":
            provider = LinearBasisProvider(
                input_dim=kwargs.pop("input_dim", 9),
                output_dim=kwargs.pop("output_dim", 6),
                include_bias=kwargs.pop("include_bias", True),
            )
            return TorchCoefficientMethod(
                provider,
                update_rule="rls",
                output_dim=provider.output_dim,
                **kwargs,
            )

        if name == "neuralfly_style_rls":
            provider = NeuralFlyStyleBasisProvider(
                input_dim=kwargs.pop("input_dim", 9),
                output_dim=kwargs.pop("output_dim", 6),
                n_basis=kwargs.pop("n_basis", 8),
                hidden_size=kwargs.pop("hidden_size", 128),
                n_hidden_layers=kwargs.pop("n_hidden_layers", 2),
            )
            return TorchCoefficientMethod(
                provider,
                update_rule="rls",
                output_dim=provider.output_dim,
                **kwargs,
            )

        if name in {"fe_rls", "fe_bayes", "fe_kalman", "fe_sgd", "fe_window_ls"}:
            try:
                model = kwargs.pop("model")
            except KeyError as exc:
                raise ValueError(f"{name} requires a loaded FunctionEncoder model") from exc

            update_rules = {
                "fe_rls": "rls",
                "fe_bayes": "kalman",
                "fe_kalman": "kalman",
                "fe_sgd": "sgd",
                "fe_window_ls": "window_ls",
            }
            if name == "fe_bayes":
                kwargs["process_noise"] = 0.0
            return TorchCoefficientMethod(
                FunctionEncoderBasisProvider(model),
                update_rule=update_rules[name],
                **kwargs,
            )

        raise NotImplementedError(f"No runtime builder is available for '{name}'")

    def _iter_point_observations(
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if features.ndim < 3:
            raise ValueError(
                f"features must have shape [batch, ..., output_dim, n_coeff], got {features.shape}"
            )
        if features.shape[:-1] != target.shape:
            raise ValueError(
                "features and target dimensions must match except coefficient axis: "
                f"{tuple(features.shape[:-1])} != {tuple(target.shape)}"
            )

        if features.ndim == 3:
            return [(features, target)]

        batch_size = features.shape[0]
        output_dim = features.shape[-2]
        n_coeff = features.shape[-1]
        feature_points = features.reshape(batch_size, -1, output_dim, n_coeff)
        target_points = target.reshape(batch_size, -1, output_dim)
        return [
            (feature_points[:, idx], target_points[:, idx])
            for idx in range(feature_points.shape[1])
        ]

    def _state_coefficients(state: CoefficientState) -> torch.Tensor:
        return state.coefficients

    def _expand_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(f"Cannot expand batch {tensor.shape[0]} to {batch_size}")
        return tensor.expand(batch_size, *tensor.shape[1:]).clone()

    def _batch_eye(batch_size: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, dtype=reference.dtype, device=reference.device)
        return eye.unsqueeze(0).expand(batch_size, dim, dim).clone()

    def _as_tensor(
        value: object,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=device)
        return torch.as_tensor(value, dtype=dtype, device=device)
