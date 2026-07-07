import unittest

try:
    import torch

    from terrain_adaptation_rls.methods.runtime import NeuralFlyStyleBasisProvider
    from terrain_adaptation_rls.training.neuralfly import (
        neuralfly_style_loss,
        predict_neuralfly_style_batch,
    )
    from terrain_adaptation_rls.training.supervised import synthetic_batch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class NeuralFlyTrainingTests(unittest.TestCase):
    def test_predict_neuralfly_style_batch_shape(self):
        model = NeuralFlyStyleBasisProvider(
            input_dim=9,
            output_dim=6,
            n_basis=3,
            hidden_size=8,
        )
        batch = synthetic_batch(
            batch_size=2,
            n_points=5,
            n_example_points=7,
            device="cpu",
        )

        prediction = predict_neuralfly_style_batch(model, batch, ridge=1e-4)

        self.assertEqual(tuple(prediction.shape), (2, 5, 6))

    def test_neuralfly_style_loss_is_scalar(self):
        model = NeuralFlyStyleBasisProvider(
            input_dim=9,
            output_dim=6,
            n_basis=3,
            hidden_size=8,
        )
        batch = synthetic_batch(
            batch_size=2,
            n_points=5,
            n_example_points=7,
            device="cpu",
        )

        loss = neuralfly_style_loss(model, batch, ridge=1e-4)

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
