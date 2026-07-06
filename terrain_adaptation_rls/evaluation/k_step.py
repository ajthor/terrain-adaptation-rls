"""Generic k-step lookahead evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from terrain_adaptation_rls.methods.protocols import RuntimeInput, RuntimeMethod


@dataclass(frozen=True)
class KStepWindow:
    """One open-loop rollout window."""

    initial_state: object
    controls: Sequence[object]
    dts: Sequence[object]
    targets: Sequence[object]
    adaptation_state: object | None = None
    time: float | None = None


@dataclass(frozen=True)
class KStepRecord:
    """Result from one k-step rollout window."""

    index: int
    time: float | None
    predictions: tuple[object, ...]
    rolled_states: tuple[object, ...]
    step_errors: tuple[float, ...]

    @property
    def accumulated_error(self) -> float:
        return sum(self.step_errors)


def run_k_step_evaluation(
    method: RuntimeMethod,
    windows: Iterable[KStepWindow],
    *,
    build_inputs: Callable[[object, object, object], RuntimeInput],
    rollout_update: Callable[[object, object], object],
    distance_fn: Callable[[object, object], float],
) -> list[KStepRecord]:
    """Evaluate recursive k-step rollouts.

    The online adaptation state is held fixed within each rollout window. This
    matches the MPPI use case: future candidate trajectories are rolled out from
    the currently adapted model before future measurements are available.
    """

    records: list[KStepRecord] = []
    default_adaptation_state = method.initial_state()

    for window_index, window in enumerate(windows):
        _check_window_lengths(window)
        adaptation_state = (
            default_adaptation_state
            if window.adaptation_state is None
            else window.adaptation_state
        )
        current_state = window.initial_state
        predictions: list[object] = []
        rolled_states: list[object] = []
        step_errors: list[float] = []

        for control, dt, target in zip(window.controls, window.dts, window.targets):
            inputs = build_inputs(current_state, control, dt)
            prediction = method.predict(adaptation_state, inputs)
            current_state = rollout_update(current_state, prediction)
            predictions.append(prediction)
            rolled_states.append(current_state)
            step_errors.append(distance_fn(target, current_state))

        records.append(
            KStepRecord(
                index=window_index,
                time=window.time,
                predictions=tuple(predictions),
                rolled_states=tuple(rolled_states),
                step_errors=tuple(step_errors),
            )
        )

    return records


def _check_window_lengths(window: KStepWindow) -> None:
    lengths = {len(window.controls), len(window.dts), len(window.targets)}
    if len(lengths) != 1:
        raise ValueError(
            "controls, dts, and targets must have the same length: "
            f"{len(window.controls)}, {len(window.dts)}, {len(window.targets)}"
        )
