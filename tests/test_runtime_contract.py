import unittest

from terrain_adaptation_rls.dynamics.runtime import RuntimeSpec, validate_runtime_shapes


class ArrayLike:
    def __init__(self, shape):
        self.shape = shape


class RuntimeContractTests(unittest.TestCase):
    def test_accepts_single_step_batch(self):
        validate_runtime_shapes(
            ArrayLike((4, 8)),
            ArrayLike((4,)),
            ArrayLike((4, 6)),
        )

    def test_accepts_horizon_batch(self):
        validate_runtime_shapes(
            ArrayLike((4, 15, 8)),
            ArrayLike((4, 15)),
            ArrayLike((4, 15, 6)),
        )

    def test_rejects_wrong_input_dim(self):
        with self.assertRaises(ValueError):
            validate_runtime_shapes(ArrayLike((4, 9)), ArrayLike((4,)))

    def test_rejects_mismatched_dt_shape(self):
        with self.assertRaises(ValueError):
            validate_runtime_shapes(ArrayLike((4, 15, 8)), ArrayLike((4,)))

    def test_supports_custom_spec(self):
        spec = RuntimeSpec(state_dim=3, control_dim=1, output_dim=3)
        validate_runtime_shapes(
            ArrayLike((2, 4)),
            ArrayLike((2,)),
            ArrayLike((2, 3)),
            spec=spec,
        )


if __name__ == "__main__":
    unittest.main()
