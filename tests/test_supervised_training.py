import unittest

from terrain_adaptation_rls.configuration import ExperimentConfig

try:
    import torch

    from terrain_adaptation_rls.training.supervised import (
        build_model_from_config,
        run_synthetic_supervised_training,
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


if __name__ == "__main__":
    unittest.main()
