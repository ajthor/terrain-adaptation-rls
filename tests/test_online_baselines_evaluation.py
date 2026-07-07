import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.online_baselines import (
        summarize_baseline_predictions,
    )
except ModuleNotFoundError:
    torch = None


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


if __name__ == "__main__":
    unittest.main()
