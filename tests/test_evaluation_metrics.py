import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.metrics import (
        summarize_logged_k_step_metrics,
        summarize_prediction_metrics,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationMetricTests(unittest.TestCase):
    def test_summarize_prediction_metrics_reports_integrated_position_error(self):
        target = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        prediction = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        zero_metrics = summarize_prediction_metrics(
            target=target,
            prediction=torch.zeros_like(target),
        )

        metrics = summarize_prediction_metrics(
            target=target,
            prediction=prediction,
            zero_metrics=zero_metrics,
        )

        self.assertAlmostEqual(metrics["mean_error"], 0.5)
        self.assertAlmostEqual(metrics["integrated_position_final_error"], 1.0)
        self.assertAlmostEqual(metrics["prediction_to_target_path_length_ratio"], 0.5)
        self.assertAlmostEqual(metrics["mean_error_to_zero_delta_ratio"], 0.5)
        self.assertIn("x_rmse", metrics)

    def test_summarize_logged_k_step_metrics_accumulates_logged_deltas(self):
        target = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        prediction = torch.zeros_like(target)

        metrics = summarize_logged_k_step_metrics(
            target=target,
            prediction=prediction,
            horizons=(1, 2),
        )

        self.assertEqual(metrics["logged_k1_n_windows"], 3.0)
        self.assertEqual(metrics["logged_k2_n_windows"], 2.0)
        self.assertAlmostEqual(metrics["logged_k1_endpoint_error_mean"], 1.0)
        self.assertAlmostEqual(metrics["logged_k2_endpoint_error_mean"], 2.0)
        self.assertAlmostEqual(metrics["logged_k2_accumulated_error_mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
