"""Generic streaming evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from terrain_adaptation_rls.methods.protocols import Observation, RuntimeMethod


@dataclass(frozen=True)
class StreamingRecord:
    """One prediction-before-update record from streaming evaluation."""

    index: int
    time: float | None
    prediction: object
    target: object
    error: float
    state_before: object
    state_after: object


def run_streaming_evaluation(
    method: RuntimeMethod,
    observations: Iterable[Observation],
    metric_fn: Callable[[object, object], float],
    *,
    initial_state: object | None = None,
) -> list[StreamingRecord]:
    """Run a method through a sequence of observations.

    The method always predicts before seeing the current target, then updates
    from that target. This ordering is the core fairness constraint for online
    adaptation comparisons.
    """

    state = method.initial_state() if initial_state is None else initial_state
    records: list[StreamingRecord] = []

    for index, observation in enumerate(observations):
        state_before = state
        prediction = method.predict(state_before, observation.inputs)
        error = metric_fn(prediction, observation.target)
        state = method.update(state_before, observation)
        records.append(
            StreamingRecord(
                index=index,
                time=observation.time,
                prediction=prediction,
                target=observation.target,
                error=error,
                state_before=state_before,
                state_after=state,
            )
        )

    return records
