import unittest

try:
    import torch

    from terrain_adaptation_rls.models.rk4 import rk4_delta_step, rk4_state_step, rk4_step
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class RK4Tests(unittest.TestCase):
    def test_delta_step_returns_delta_not_next_state(self):
        def constant_dynamics(t, x):
            return torch.ones_like(x[..., :6]) * 2.0

        x = torch.zeros(1, 8)
        dt = torch.tensor([0.25])

        delta = rk4_delta_step(constant_dynamics, x, dt)

        self.assertEqual(tuple(delta.shape), (1, 6))
        self.assertTrue(torch.allclose(delta, torch.full((1, 6), 0.5)))

    def test_delta_step_advances_state_in_substeps(self):
        def linear_dynamics(t, x):
            return x[..., :6]

        x = torch.cat([torch.ones(1, 6), torch.zeros(1, 2)], dim=-1)
        dt = torch.tensor([1.0])

        delta = rk4_delta_step(linear_dynamics, x, dt)

        rk4_expected = torch.full((1, 6), 1.0 + 0.5 + 1.0 / 6.0 + 1.0 / 24.0)
        euler_like = torch.ones(1, 6)
        self.assertTrue(torch.allclose(delta, rk4_expected, atol=1e-6))
        self.assertFalse(torch.allclose(delta, euler_like))

    def test_state_step_adds_delta_and_preserves_controls(self):
        def constant_dynamics(t, x):
            return torch.ones_like(x[..., :6])

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3, -0.4]])
        dt = torch.tensor([0.5])

        next_state = rk4_state_step(constant_dynamics, x, dt)

        self.assertTrue(torch.allclose(next_state[..., :6], x[..., :6] + 0.5))
        self.assertTrue(torch.allclose(next_state[..., 6:8], x[..., 6:8]))

    def test_legacy_name_is_delta_step(self):
        self.assertIs(rk4_step, rk4_delta_step)


if __name__ == "__main__":
    unittest.main()
