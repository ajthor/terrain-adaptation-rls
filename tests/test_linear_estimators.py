import unittest

try:
    import torch

    from terrain_adaptation_rls.estimators.linear import (
        CoefficientSGDState,
        KalmanState,
        RLSState,
        WindowedLeastSquaresState,
        coefficient_sgd_update,
        kalman_update,
        linear_predict,
        rls_update,
        solve_ridge_coefficients,
        windowed_least_squares_update,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class LinearEstimatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, -1.0]],
            ]
        )
        self.coefficients = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        self.targets = linear_predict(self.features, self.coefficients)

    def test_linear_predict(self):
        prediction = linear_predict(self.features, self.coefficients)
        self.assertTrue(torch.allclose(prediction, self.targets))

    def test_rls_update_recovers_single_observation_when_directly_observed(self):
        state = RLSState(
            coefficients=torch.zeros(1, 2),
            covariance=1000.0 * torch.eye(2).unsqueeze(0),
        )
        features = torch.eye(2).unsqueeze(0)
        target = torch.tensor([[2.0, -1.0]])

        updated = rls_update(state, features, target, measurement_noise=1e-9)

        self.assertTrue(torch.allclose(updated.coefficients, target, atol=1e-5))

    def test_kalman_update_recovers_single_observation_when_directly_observed(self):
        state = KalmanState(
            coefficients=torch.zeros(1, 2),
            covariance=1000.0 * torch.eye(2).unsqueeze(0),
        )
        features = torch.eye(2).unsqueeze(0)
        target = torch.tensor([[2.0, -1.0]])

        updated = kalman_update(state, features, target, measurement_noise=1e-9)

        self.assertTrue(torch.allclose(updated.coefficients, target, atol=1e-5))

    def test_coefficient_sgd_reduces_error(self):
        state = CoefficientSGDState(coefficients=torch.zeros_like(self.coefficients))
        before = torch.nn.functional.mse_loss(
            linear_predict(self.features, state.coefficients),
            self.targets,
        )

        updated = coefficient_sgd_update(
            state,
            self.features,
            self.targets,
            learning_rate=0.1,
        )
        after = torch.nn.functional.mse_loss(
            linear_predict(self.features, updated.coefficients),
            self.targets,
        )

        self.assertLess(after.item(), before.item())

    def test_solve_ridge_coefficients(self):
        features = self.features.unsqueeze(1)
        targets = self.targets.unsqueeze(1)

        coefficients = solve_ridge_coefficients(features, targets, ridge=1e-9)

        self.assertTrue(torch.allclose(coefficients, self.coefficients, atol=1e-5))

    def test_windowed_least_squares_update_appends_window(self):
        state = WindowedLeastSquaresState(
            coefficients=torch.zeros_like(self.coefficients),
            features=torch.empty(2, 0, 2, 2),
            targets=torch.empty(2, 0, 2),
        )

        updated = windowed_least_squares_update(
            state,
            self.features,
            self.targets,
            window_size=2,
            ridge=1e-9,
        )

        self.assertEqual(updated.features.shape[1], 1)
        self.assertTrue(torch.allclose(updated.coefficients, self.coefficients, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
