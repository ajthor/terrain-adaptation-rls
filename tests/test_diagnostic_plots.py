import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.diagnostic_plots import (
        integrate_planar_deltas,
        quantile_range,
        summarize_trajectory_scales,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class DiagnosticPlotTests(unittest.TestCase):
    def test_integrates_planar_deltas_straight_line(self):
        deltas = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        poses = integrate_planar_deltas(deltas)

        self.assertTrue(torch.allclose(poses[:, 0], torch.tensor([0.0, 1.0, 3.0])))
        self.assertTrue(torch.allclose(poses[:, 1], torch.zeros(3)))

    def test_integrates_planar_deltas_with_yaw(self):
        deltas = torch.tensor(
            [
                [0.0, 0.0, torch.pi / 2, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        poses = integrate_planar_deltas(deltas)

        self.assertAlmostEqual(poses[-1, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(poses[-1, 1].item(), 1.0, places=5)

    def test_quantile_range_expands_degenerate_values(self):
        lower, upper = quantile_range(torch.ones(10))

        self.assertLess(lower, 1.0)
        self.assertGreater(upper, 1.0)

    def test_summarizes_tiny_prediction_scale(self):
        target = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            ]
        ).transpose(0, 1)
        prediction = torch.zeros_like(target)
        dt = torch.full((1, 2), 0.1)

        summary = summarize_trajectory_scales(
            target=target,
            prediction=prediction,
            dt=dt,
            scene="debug",
        )

        self.assertEqual(summary["n_steps"], 2)
        self.assertEqual(summary["trajectory"]["target"]["path_length"], 2.0)
        self.assertEqual(summary["trajectory"]["prediction"]["path_length"], 0.0)
        self.assertEqual(summary["delta_norms"]["prediction_to_target_planar_ratio"], 0.0)
        self.assertEqual(summary["delta_norms"]["prediction_error_to_zero_delta_error_ratio"], 1.0)
        self.assertIn(
            "prediction_planar_deltas_are_less_than_10_percent_of_target",
            summary["flags"],
        )
        self.assertIn(
            "prediction_error_is_within_5_percent_of_zero_delta_baseline",
            summary["flags"],
        )


if __name__ == "__main__":
    unittest.main()
