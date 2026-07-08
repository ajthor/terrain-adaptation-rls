import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.online_baselines import (
        summarize_baseline_predictions,
        summarize_recursive_k_step_metrics,
    )
except ModuleNotFoundError:
    torch = None


class ZeroPredictor:
    def predict(self, *, state, control, dt, start_index):
        del state, control, dt, start_index
        return torch.zeros(6)


@unittest.skipIf(torch is None, "torch is not installed")
class OnlineBaselineEvaluationTests(unittest.TestCase):
    def test_summarize_baseline_predictions_reports_zero_ratio(self):
        target = torch.ones(1, 4, 6)
        predictions = {
            "good": target.clone(),
            "zero_delta": torch.zeros_like(target),
        }

        summary = summarize_baseline_predictions(
            target=target,
            predictions=predictions,
            coefficient_histories={},
            scene="scene_test",
            forgetting_factor=0.95,
            initial_covariance=1000.0,
            measurement_noise=1e-6,
            start_index=0,
            n_example_points=2,
            linear_include_bias=True,
        )

        self.assertEqual(summary["scene"], "scene_test")
        self.assertEqual(summary["methods"]["good"]["mean_error"], 0.0)
        self.assertEqual(
            summary["methods"]["zero_delta"]["mean_error_to_zero_delta_ratio"],
            1.0,
        )

    def test_summarize_recursive_k_step_metrics_uses_legacy_rollout_targets(self):
        xs = torch.zeros(1, 4, 8)
        dt = torch.ones(1, 4)
        target = torch.ones(1, 4, 6)

        summary = summarize_recursive_k_step_metrics(
            xs=xs,
            dt=dt,
            target=target,
            predictors={"zero": ZeroPredictor()},
            horizons=(1, 2),
            max_rollouts=2,
        )

        self.assertEqual(summary["zero"]["recursive_k1_n_rollouts"], 2.0)
        self.assertEqual(summary["zero"]["recursive_k2_n_rollouts"], 2.0)
        self.assertAlmostEqual(
            summary["zero"]["recursive_k1_final_step_error_mean"],
            6**0.5,
            places=6,
        )
        self.assertAlmostEqual(
            summary["zero"]["recursive_k2_accumulated_error_mean"],
            2 * 6**0.5,
            places=6,
        )
        self.assertAlmostEqual(
            summary["zero"]["recursive_k2_trajectory_rmse_mean"],
            6**0.5,
            places=6,
        )
        self.assertAlmostEqual(
            summary["zero"]["recursive_k2_integral_square_error_mean"],
            12.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
