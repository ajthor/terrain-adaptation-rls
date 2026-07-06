import unittest

try:
    import torch

    from terrain_adaptation_rls.estimators.linear import linear_predict
    from terrain_adaptation_rls.methods.builders import build_runtime_method
    from terrain_adaptation_rls.methods.coefficient_adapters import (
        LinearBasisProvider,
        NeuralFlyStyleBasisProvider,
        TorchCoefficientMethod,
        concatenate_runtime_input,
    )
    from terrain_adaptation_rls.methods.protocols import Observation, RuntimeInput
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class RuntimeMethodAdapterTests(unittest.TestCase):
    def test_concatenates_runtime_input(self):
        inputs = RuntimeInput(
            xs=torch.ones(2, 3, 8),
            dt=torch.full((2, 3), 0.1),
        )

        z = concatenate_runtime_input(inputs)

        self.assertEqual(tuple(z.shape), (2, 3, 9))
        self.assertTrue(torch.allclose(z[..., -1], torch.full((2, 3), 0.1)))

    def test_linear_basis_provider_is_block_diagonal(self):
        provider = LinearBasisProvider(input_dim=3, output_dim=2)
        inputs = RuntimeInput(
            xs=torch.tensor([[2.0, -1.0]]),
            dt=torch.tensor([0.5]),
        )

        features = provider(inputs)

        self.assertEqual(tuple(features.shape), (1, 2, 8))
        expected_scalar_features = torch.tensor([2.0, -1.0, 0.5, 1.0])
        self.assertTrue(torch.allclose(features[0, 0, :4], expected_scalar_features))
        self.assertTrue(torch.allclose(features[0, 0, 4:], torch.zeros(4)))
        self.assertTrue(torch.allclose(features[0, 1, :4], torch.zeros(4)))
        self.assertTrue(torch.allclose(features[0, 1, 4:], expected_scalar_features))

    def test_rls_method_fits_current_linear_basis_observation(self):
        provider = LinearBasisProvider(input_dim=3, output_dim=2)
        method = TorchCoefficientMethod(
            provider,
            update_rule="rls",
            initial_covariance=1e6,
            measurement_noise=1e-9,
        )
        inputs = RuntimeInput(
            xs=torch.tensor([[2.0, -1.0]]),
            dt=torch.tensor([0.5]),
        )
        target = torch.tensor([[1.25, -0.75]])

        state = method.initial_state()
        before = method.predict(state, inputs)
        updated = method.update(state, Observation(inputs=inputs, target=target))
        after = method.predict(updated, inputs)

        self.assertGreater(torch.norm(before - target).item(), 1.0)
        self.assertLess(torch.norm(after - target).item(), 1e-4)

    def test_sequence_update_applies_each_point(self):
        provider = LinearBasisProvider(input_dim=1, output_dim=2, include_bias=False)
        true_coefficients = torch.tensor([[0.4, -0.3]])
        method = TorchCoefficientMethod(
            provider,
            update_rule="window_ls",
            output_dim=2,
            window_size=2,
            ridge=1e-9,
        )
        inputs = RuntimeInput(
            xs=torch.empty(1, 2, 0),
            dt=torch.tensor([[0.1, 0.2]]),
        )
        features = provider(inputs)
        target = linear_predict(features, true_coefficients)

        updated = method.update(method.initial_state(), Observation(inputs=inputs, target=target))

        self.assertEqual(tuple(updated.features.shape), (1, 2, 2, provider.n_coeff))
        self.assertTrue(torch.allclose(method.predict(updated, inputs), target, atol=1e-5))

    def test_neuralfly_style_basis_provider_shape(self):
        provider = NeuralFlyStyleBasisProvider(
            input_dim=9,
            output_dim=6,
            n_basis=8,
            hidden_size=16,
        )
        inputs = RuntimeInput(
            xs=torch.zeros(2, 4, 8),
            dt=torch.full((2, 4), 0.05),
        )

        features = provider(inputs)

        self.assertEqual(tuple(features.shape), (2, 4, 6, 8))

    def test_builder_creates_linear_rls_method(self):
        method = build_runtime_method("linear_basis_rls", input_dim=3, output_dim=2)

        self.assertIsInstance(method, TorchCoefficientMethod)
        self.assertEqual(method.n_coeff, 8)


if __name__ == "__main__":
    unittest.main()
