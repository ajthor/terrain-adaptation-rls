"""Runtime shape contract for Phoenix-compatible dynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSpec:
    """Shape convention for terrain dynamics models."""

    state_dim: int = 6
    control_dim: int = 2
    output_dim: int = 6

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.control_dim


def validate_runtime_shapes(
    xs: Any,
    dt: Any,
    prediction: Any | None = None,
    *,
    spec: RuntimeSpec = RuntimeSpec(),
) -> None:
    """Validate Phoenix-shaped runtime tensors/arrays.

    Accepts either single-step batches:

    ```text
    xs: [batch, input_dim]
    dt: [batch]
    prediction: [batch, output_dim]
    ```

    or horizon batches:

    ```text
    xs: [batch, horizon, input_dim]
    dt: [batch, horizon]
    prediction: [batch, horizon, output_dim]
    ```
    """

    xs_shape = _shape(xs, "xs")
    dt_shape = _shape(dt, "dt")
    if len(xs_shape) not in (2, 3):
        raise ValueError(f"xs must be rank 2 or 3, got shape {xs_shape}")
    if xs_shape[-1] != spec.input_dim:
        raise ValueError(
            f"xs last dimension must be {spec.input_dim}, got shape {xs_shape}"
        )
    expected_dt_shape = xs_shape[:-1]
    if dt_shape != expected_dt_shape:
        raise ValueError(
            f"dt shape must match xs batch/horizon dimensions: "
            f"expected {expected_dt_shape}, got {dt_shape}"
        )

    if prediction is not None:
        prediction_shape = _shape(prediction, "prediction")
        expected_prediction_shape = (*expected_dt_shape, spec.output_dim)
        if prediction_shape != expected_prediction_shape:
            raise ValueError(
                f"prediction shape must be {expected_prediction_shape}, "
                f"got {prediction_shape}"
            )


def _shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{name} must expose a shape attribute")
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError(f"{name}.shape must be iterable") from exc
