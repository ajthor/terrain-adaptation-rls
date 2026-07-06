import unittest

from terrain_adaptation_rls.evaluation.k_step import KStepWindow, run_k_step_evaluation
from terrain_adaptation_rls.methods.protocols import RuntimeInput


class IncrementMethod:
    def initial_state(self):
        return 0.0

    def predict(self, state, inputs):
        return inputs.xs + state

    def update(self, state, observation):
        raise AssertionError("k-step evaluation should not update online state")


class KStepEvaluationTests(unittest.TestCase):
    def test_rollout_holds_adaptation_state_fixed(self):
        windows = [
            KStepWindow(
                initial_state=0.0,
                controls=(1.0, 1.0),
                dts=(0.1, 0.1),
                targets=(2.0, 5.0),
                adaptation_state=1.0,
                time=0.0,
            )
        ]

        records = run_k_step_evaluation(
            IncrementMethod(),
            windows,
            build_inputs=lambda state, control, dt: RuntimeInput(xs=state + control, dt=dt),
            rollout_update=lambda state, prediction: state + prediction,
            distance_fn=lambda target, state: abs(target - state),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].predictions, (2.0, 4.0))
        self.assertEqual(records[0].rolled_states, (2.0, 6.0))
        self.assertEqual(records[0].step_errors, (0.0, 1.0))
        self.assertEqual(records[0].accumulated_error, 1.0)

    def test_rejects_mismatched_window_lengths(self):
        windows = [
            KStepWindow(
                initial_state=0.0,
                controls=(1.0,),
                dts=(0.1, 0.1),
                targets=(1.0,),
            )
        ]

        with self.assertRaises(ValueError):
            run_k_step_evaluation(
                IncrementMethod(),
                windows,
                build_inputs=lambda state, control, dt: RuntimeInput(xs=state, dt=dt),
                rollout_update=lambda state, prediction: prediction,
                distance_fn=lambda target, state: 0.0,
            )


if __name__ == "__main__":
    unittest.main()
