import unittest

try:
    import torch

    from terrain_adaptation_rls.evaluation.fe_rls import scene_streaming_tensors
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class FERLSEvaluationTests(unittest.TestCase):
    def test_scene_streaming_tensors_builds_delta_targets(self):
        inputs = torch.zeros(6, 9)
        targets = torch.zeros(6, 7)
        inputs[:, 0] = torch.arange(6, dtype=torch.float32)
        targets[:, 0] = inputs[:, 0] + 0.1
        inputs[:, 1:7] = 1.0
        targets[:, 1:7] = 3.0

        xs, dt, target, time = scene_streaming_tensors(
            inputs=inputs,
            targets=targets,
            start_index=1,
            max_points=3,
            device="cpu",
        )

        self.assertEqual(tuple(xs.shape), (1, 3, 8))
        self.assertTrue(torch.allclose(dt, torch.full((1, 3), 0.1)))
        self.assertTrue(torch.allclose(target, torch.full((1, 3, 6), 2.0)))
        self.assertTrue(torch.allclose(time, torch.tensor([0.0, 1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
