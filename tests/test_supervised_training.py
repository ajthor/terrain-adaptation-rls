import unittest

from terrain_adaptation_rls.configuration import ExperimentConfig

try:
    import torch

    from terrain_adaptation_rls.training.supervised import (
        build_model_from_config,
        run_synthetic_supervised_training,
        scene_sequence_batch,
        scene_trajectory_batch,
        synthetic_batch,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch/function_encoder is not installed")
class SupervisedTrainingTests(unittest.TestCase):
    def test_builds_neural_ode_from_config(self):
        config = ExperimentConfig(
            name="node",
            kind="train",
            model={"family": "neural_ode", "hidden_size": 8, "n_basis": 2},
        )

        built = build_model_from_config(config, device="cpu")

        self.assertEqual(built.family, "neural_ode")

    def test_synthetic_batch_shapes(self):
        batch = synthetic_batch(
            batch_size=2,
            n_points=3,
            n_example_points=4,
            device="cpu",
        )

        self.assertEqual(tuple(batch[0].shape), (2, 3, 8))
        self.assertEqual(tuple(batch[1].shape), (2, 3))
        self.assertEqual(tuple(batch[2].shape), (2, 3, 6))
        self.assertEqual(tuple(batch[3].shape), (2, 4, 8))

    def test_runs_synthetic_training(self):
        config = ExperimentConfig(
            name="node",
            kind="train",
            seed=3,
            model={"family": "neural_ode", "hidden_size": 8, "n_basis": 2},
            training={"steps": 2, "batch_size": 2, "n_points": 3},
        )

        metrics = run_synthetic_supervised_training(config, device="cpu")

        self.assertEqual(metrics["family"], "neural_ode")
        self.assertEqual(metrics["steps"], 2)
        self.assertEqual(len(metrics["losses"]), 2)

    def test_maml_training_reports_missing_port(self):
        config = ExperimentConfig(
            name="maml",
            kind="train",
            model={"family": "maml_neural_ode"},
        )

        with self.assertRaisesRegex(NotImplementedError, "MAML meta-training"):
            build_model_from_config(config, device="cpu")

    def test_scene_sequence_batch_keeps_query_order(self):
        inputs = torch.zeros(10, 9)
        targets = torch.zeros(10, 7)
        inputs[:, 0] = torch.arange(10)
        targets[:, 0] = torch.arange(10) + 0.1
        targets[:, 1] = torch.arange(10)

        batch = scene_sequence_batch(
            inputs=inputs,
            targets=targets,
            n_example_points=2,
            max_query_points=3,
            device="cpu",
        )

        xs, dt, ys, example_xs, *_ = batch
        self.assertEqual(tuple(xs.shape), (1, 3, 8))
        self.assertTrue(torch.allclose(dt, torch.full((1, 3), 0.1)))
        self.assertTrue(torch.allclose(ys[0, :, 0], torch.tensor([2.0, 3.0, 4.0])))
        self.assertEqual(tuple(example_xs.shape), (1, 2, 8))

    def test_scene_trajectory_batch_uses_random_examples_outside_query(self):
        inputs = torch.zeros(10, 9)
        targets = torch.zeros(10, 7)
        inputs[:, 0] = torch.arange(10)
        inputs[:, 1] = torch.arange(10)
        targets[:, 0] = torch.arange(10) + 0.1
        targets[:, 1] = torch.arange(10)

        batch = scene_trajectory_batch(
            inputs=inputs,
            targets=targets,
            n_example_points=2,
            max_query_points=3,
            device="cpu",
            query_start_index=2,
            example_policy="random_scene",
            seed=0,
        )

        xs, _, _, example_xs, *_ = batch
        self.assertTrue(torch.allclose(xs[0, :, 0], torch.tensor([2.0, 3.0, 4.0])))
        for row_id in example_xs[0, :, 0].tolist():
            self.assertNotIn(row_id, [2.0, 3.0, 4.0])

    def test_scene_trajectory_batch_can_use_preceding_examples(self):
        inputs = torch.zeros(10, 9)
        targets = torch.zeros(10, 7)
        inputs[:, 0] = torch.arange(10)
        inputs[:, 1] = torch.arange(10)
        targets[:, 0] = torch.arange(10) + 0.1

        batch = scene_trajectory_batch(
            inputs=inputs,
            targets=targets,
            n_example_points=2,
            max_query_points=3,
            device="cpu",
            query_start_index=4,
            example_policy="preceding",
        )

        xs, _, _, example_xs, *_ = batch
        self.assertTrue(torch.allclose(example_xs[0, :, 0], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(xs[0, :, 0], torch.tensor([4.0, 5.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
