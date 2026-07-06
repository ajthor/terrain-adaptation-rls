import unittest

from terrain_adaptation_rls.evaluation.streaming import run_streaming_evaluation
from terrain_adaptation_rls.methods.protocols import Observation, RuntimeInput


class AdditiveMethod:
    def initial_state(self):
        return 0.0

    def predict(self, state, inputs):
        return inputs.xs + state

    def update(self, state, observation):
        return observation.target - observation.inputs.xs


class StreamingEvaluationTests(unittest.TestCase):
    def test_predicts_before_update(self):
        observations = [
            Observation(RuntimeInput(xs=1.0, dt=0.1), target=3.0, time=0.0),
            Observation(RuntimeInput(xs=2.0, dt=0.1), target=7.0, time=0.1),
        ]

        records = run_streaming_evaluation(
            AdditiveMethod(),
            observations,
            metric_fn=lambda prediction, target: abs(target - prediction),
        )

        self.assertEqual(records[0].prediction, 1.0)
        self.assertEqual(records[0].state_after, 2.0)
        self.assertEqual(records[1].state_before, 2.0)
        self.assertEqual(records[1].prediction, 4.0)
        self.assertEqual(records[1].state_after, 5.0)
        self.assertEqual([record.error for record in records], [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
