import unittest

try:
    import torch

    from terrain_adaptation_rls.methods.runtime import ALPaCABasisProvider, RuntimeInput
    from terrain_adaptation_rls.training.alpaca import (
        alpaca_loss,
        alpaca_posterior_from_context,
        alpaca_prior_posterior,
        alpaca_update_posterior,
        alpaca_zero_posterior,
        predict_alpaca_batch,
    )
    from terrain_adaptation_rls.training.supervised import synthetic_batch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ALPaCATrainingTests(unittest.TestCase):
    def test_predict_alpaca_batch_shape(self):
        model = ALPaCABasisProvider(
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

        prediction = predict_alpaca_batch(model, batch)

        self.assertEqual(tuple(prediction.shape), (2, 5, 6))

    def test_alpaca_loss_is_scalar(self):
        model = ALPaCABasisProvider(
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

        loss = alpaca_loss(model, batch)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_posterior_update_changes_mean(self):
        model = ALPaCABasisProvider(
            input_dim=9,
            output_dim=6,
            n_basis=3,
            hidden_size=8,
        )
        batch = synthetic_batch(
            batch_size=1,
            n_points=5,
            n_example_points=7,
            device="cpu",
        )
        _, _, _, example_xs, example_dt, example_ys = batch

        context_features = model(RuntimeInput(example_xs, example_dt))
        posterior = alpaca_posterior_from_context(model, context_features, example_ys)
        prior = alpaca_prior_posterior(model, batch_size=1)
        updated = alpaca_update_posterior(
            model,
            prior,
            context_features[:, :1],
            example_ys[:, :1],
        )

        self.assertEqual(tuple(posterior.mean.shape), (1, 3))
        self.assertEqual(tuple(posterior.covariance.shape), (1, 3, 3))
        self.assertGreater(torch.norm(updated.mean - prior.mean).item(), 0.0)

    def test_zero_posterior_starts_from_zero_coefficients(self):
        model = ALPaCABasisProvider(
            input_dim=9,
            output_dim=6,
            n_basis=3,
            hidden_size=8,
        )
        batch = synthetic_batch(
            batch_size=1,
            n_points=5,
            n_example_points=7,
            device="cpu",
        )
        _, _, _, example_xs, example_dt, example_ys = batch

        posterior = alpaca_zero_posterior(
            model,
            batch_size=1,
            initial_covariance=25.0,
        )
        updated = alpaca_update_posterior(
            model,
            posterior,
            model(RuntimeInput(example_xs[:, :1], example_dt[:, :1])),
            example_ys[:, :1],
        )

        self.assertTrue(torch.allclose(posterior.mean, torch.zeros_like(posterior.mean)))
        self.assertTrue(
            torch.allclose(
                torch.diagonal(posterior.covariance, dim1=-2, dim2=-1),
                torch.full((1, 3), 25.0),
            )
        )
        self.assertGreater(torch.norm(updated.mean - posterior.mean).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
