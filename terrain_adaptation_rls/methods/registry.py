"""Method registry metadata.

This registry names the method families expected by configs and evaluation
commands without importing model implementations. Actual builders can be added
later once the devcontainer has the FE/MAML dependencies available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    """Metadata for a train/eval method."""

    name: str
    category: str
    phoenix_compatible: bool
    requires_training: bool
    requires_torch: bool
    description: str


DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="node_static",
        category="static",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Static neural ODE dynamics model.",
    ),
    MethodSpec(
        name="fe_static",
        category="static",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder with batch/offline coefficients.",
    ),
    MethodSpec(
        name="fe_rls",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with recursive least squares coefficients.",
    ),
    MethodSpec(
        name="fe_bayes",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with Bayesian linear coefficient updates.",
    ),
    MethodSpec(
        name="fe_prior_bayes",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with training-scene prior coefficients and Bayesian updates.",
    ),
    MethodSpec(
        name="maml_online",
        category="gradient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="MAML initialization adapted online with gradient steps.",
    ),
    MethodSpec(
        name="fe_sgd",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with online coefficient SGD.",
    ),
    MethodSpec(
        name="fe_window_ls",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with sliding-window ridge least squares.",
    ),
    MethodSpec(
        name="fe_kalman",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="Function Encoder basis with Kalman-style coefficient updates.",
    ),
    MethodSpec(
        name="linear_basis_rls",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=False,
        requires_torch=True,
        description="Hand-designed linear/identity basis with RLS coefficients.",
    ),
    MethodSpec(
        name="neuralfly_style_rls",
        category="coefficient_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="NeuralFly-style learned basis with online low-dimensional adaptation.",
    ),
    MethodSpec(
        name="alpaca_online",
        category="bayesian_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="ALPaCA learned features and Bayesian online last-layer adaptation.",
    ),
    MethodSpec(
        name="alpaca_cold_start_online",
        category="bayesian_adaptation",
        phoenix_compatible=True,
        requires_training=True,
        requires_torch=True,
        description="ALPaCA learned features with zero-mean Bayesian online last-layer adaptation.",
    ),
)


def default_method_registry() -> dict[str, MethodSpec]:
    """Return the default registry keyed by method name."""

    return {method.name: method for method in DEFAULT_METHODS}


def get_method_spec(name: str, registry: dict[str, MethodSpec] | None = None) -> MethodSpec:
    """Get one method spec by name."""

    registry = default_method_registry() if registry is None else registry
    try:
        return registry[name]
    except KeyError as exc:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown method '{name}'. Known methods: {known}") from exc


def validate_method_names(
    names: list[str] | tuple[str, ...],
    registry: dict[str, MethodSpec] | None = None,
) -> tuple[MethodSpec, ...]:
    """Validate method names from a config."""

    return tuple(get_method_spec(name, registry=registry) for name in names)
