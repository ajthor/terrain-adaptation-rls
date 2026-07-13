import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.metrics import (
        summarize_adaptation_time_metrics,
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
        self.assertAlmostEqual(metrics["logged_k2_trajectory_rmse_mean"], (2.5) ** 0.5)
        self.assertAlmostEqual(metrics["logged_k2_integral_square_error_mean"], 5.0)

    def test_summarize_adaptation_time_metrics_reports_threshold_crossing(self):
        errors = torch.tensor([10.0, 10.0, 8.0, 6.0, 4.0])
        dt = torch.full((5,), 0.1)

        metrics = summarize_adaptation_time_metrics(
            errors,
            dt=dt,
            window=2,
            improvement_thresholds=(0.25, 0.50),
        )

        self.assertEqual(metrics["adaptation_window"], 2.0)
        self.assertAlmostEqual(metrics["adaptation_initial_error"], 10.0)
        self.assertEqual(metrics["adaptation_samples_to_25pct_improvement"], 4.0)
        self.assertAlmostEqual(metrics["adaptation_seconds_to_25pct_improvement"], 0.4)
        self.assertEqual(metrics["adaptation_reached_25pct_improvement"], 1.0)
        self.assertEqual(metrics["adaptation_samples_to_50pct_improvement"], 5.0)

    def test_summarize_adaptation_time_metrics_penalizes_never_crossing(self):
        errors = torch.tensor([10.0, 9.5, 9.0])

        metrics = summarize_adaptation_time_metrics(
            errors,
            window=1,
            improvement_thresholds=(0.50,),
        )

        self.assertEqual(metrics["adaptation_samples_to_50pct_improvement"], 4.0)
        self.assertEqual(metrics["adaptation_reached_50pct_improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()
