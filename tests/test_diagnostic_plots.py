import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.diagnostic_plots import (
        integrate_planar_deltas,
        quantile_range,
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


if __name__ == "__main__":
    unittest.main()
