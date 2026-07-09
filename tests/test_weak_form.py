import unittest

try:
    import torch

    from terrain_adaptation_rls.training.weak_form import (
        sine_test_functions_from_dt,
        solve_weak_coefficients,
        weak_system_from_basis,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class WeakFormTests(unittest.TestCase):
    def test_sine_test_functions_have_expected_shape(self):
        dt = torch.full((2, 5), 0.1)

        phi, phi_prime = sine_test_functions_from_dt(dt, n_tests=3)

        self.assertEqual(tuple(phi.shape), (2, 3, 5))
        self.assertEqual(tuple(phi_prime.shape), (2, 3, 5))
        self.assertTrue(torch.allclose(phi[:, :, 0], torch.zeros(2, 3), atol=1e-6))

    def test_weak_system_recovers_constant_velocity_coefficient(self):
        dt = torch.full((1, 200), 0.01)
        time = torch.cumsum(dt, dim=-1) - dt
        state = time.unsqueeze(-1)
        basis = torch.ones(1, 200, 1, 1)

        weak_basis, weak_target = weak_system_from_basis(
            state,
            basis,
            dt,
            n_tests=5,
        )
        coefficients = solve_weak_coefficients(weak_basis, weak_target, ridge=1e-9)

        self.assertTrue(torch.allclose(coefficients, torch.ones_like(coefficients), atol=5e-2))


if __name__ == "__main__":
    unittest.main()
