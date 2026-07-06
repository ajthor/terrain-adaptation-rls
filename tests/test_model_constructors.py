import unittest

try:
    import torch

    from terrain_adaptation_rls.models.function_encoder import create_model as create_fe
    from terrain_adaptation_rls.models.maml import create_model as create_maml
    from terrain_adaptation_rls.models.neural_ode import create_model as create_node
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch/function_encoder is not installed")
class ModelConstructorTests(unittest.TestCase):
    def test_neural_ode_forward_shape(self):
        model = create_node("cpu", n_basis=2, hidden_size=8)
        xs = torch.zeros(1, 3, 8)
        dt = torch.full((1, 3), 0.1)

        prediction = model((xs, dt))

        self.assertEqual(tuple(prediction.shape), (1, 3, 6))

    def test_maml_forward_shape(self):
        model = create_maml("cpu", n_basis=2, hidden_size=8)
        xs = torch.zeros(1, 3, 8)
        dt = torch.full((1, 3), 0.1)

        prediction = model((xs, dt))

        self.assertEqual(tuple(prediction.shape), (1, 3, 6))

    def test_function_encoder_basis_and_forward_shape(self):
        model = create_fe("cpu", n_basis=2, hidden_size=8)
        xs = torch.zeros(1, 3, 8)
        dt = torch.full((1, 3), 0.1)
        coefficients = torch.zeros(1, 2)

        basis = model.basis_functions((xs, dt))
        prediction = model((xs, dt), coefficients=coefficients)

        self.assertEqual(tuple(basis.shape), (1, 3, 6, 2))
        self.assertEqual(tuple(prediction.shape), (1, 3, 6))


if __name__ == "__main__":
    unittest.main()
