"""Protocols for Phoenix-shaped runtime methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
    """Protocol for model-plus-update wrappers.

    Implementations may wrap FE, MAML, NODE, NeuralFly-style, or hand-designed
    basis models. The underlying model does not need to implement this protocol
    directly.
    """

    def initial_state(self) -> object:
        """Return the initial online adaptation state."""

    def predict(self, state: object, inputs: RuntimeInput) -> object:
        """Predict from the current state before seeing the current target."""

    def update(self, state: object, observation: Observation) -> object:
        """Update online state after the current prediction is recorded."""
