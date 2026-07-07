import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from terrain_adaptation_rls.evaluation.vanderpol_toy import (
        FEIncrementBasis,
        FEODEBasis,
        NeuralFlyToyBasis,
        generate_vanderpol_trajectory,
        run_vanderpol_toy_evaluation,
        write_seed_sweep_summary,
    )
    from terrain_adaptation_rls.methods.runtime import RuntimeInput
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class VanDerPolToyTests(unittest.TestCase):
    def test_basis_shapes_match_linear_coefficient_contract(self):
        xs = torch.zeros(2, 5, 2)
        dt = torch.full((2, 5), 0.02)

        for model in (
            FEODEBasis(n_basis=3, hidden_size=8),
            FEIncrementBasis(n_basis=3, hidden_size=8),
            NeuralFlyToyBasis(n_basis=3, hidden_size=8),
        ):
            features = model(RuntimeInput(xs, dt))
            self.assertEqual(tuple(features.shape), (2, 5, 2, 3))

    def test_vanderpol_trajectory_shapes(self):
        trajectory = generate_vanderpol_trajectory(
            mu=1.0,
            steps=10,
            dt=0.02,
            x0=torch.tensor([2.0, 0.0]),
        )

        self.assertEqual(tuple(trajectory["states"].shape), (11, 2))
        self.assertEqual(tuple(trajectory["xs"].shape), (10, 2))
        self.assertEqual(tuple(trajectory["deltas"].shape), (10, 2))
        self.assertEqual(tuple(trajectory["dt"].shape), (10,))

    def test_runs_tiny_vanderpol_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_vanderpol_toy_evaluation(
                artifact_dir=tmpdir,
                train_steps=1,
                hidden_size=8,
                n_basis=2,
                batch_size=2,
                n_example_points=4,
                n_query_points=4,
                train_mus=(0.5, 1.0),
                test_mus=(0.75,),
                train_trajectories_per_mu=2,
                train_trajectory_steps=12,
                eval_steps=20,
            )

            artifact_dir = Path(result.artifact_dir)
            self.assertTrue((artifact_dir / "method_summary.csv").exists())
            self.assertTrue((artifact_dir / "summary.json").exists())
            self.assertTrue((artifact_dir / "metric_summary_grid.png").exists())
            self.assertTrue((artifact_dir / "training_losses.png").exists())
            self.assertTrue((artifact_dir / "basis_streamplots_fe_ode_rls.png").exists())
            self.assertTrue((artifact_dir / "basis_streamplots_fe_mlp_rls.png").exists())
            self.assertTrue((artifact_dir / "basis_streamplots_neuralfly_rls.png").exists())
            self.assertTrue((artifact_dir / "interpolation_mu_0.75_component_errors.png").exists())
            self.assertTrue(
                (artifact_dir / "interpolation_mu_0.75_recursive_horizon_errors.png").exists()
            )
            self.assertTrue((artifact_dir / "interpolation_mu_0.75_rollout_snapshots.png").exists())
            self.assertTrue((artifact_dir / "interpolation_mu_0.75_streamplots.png").exists())

            write_seed_sweep_summary(artifact_dir, [result])
            self.assertTrue((artifact_dir / "seed_method_summary.csv").exists())
            self.assertTrue((artifact_dir / "aggregate_method_summary.csv").exists())
            self.assertTrue((artifact_dir / "aggregate_metric_summary.png").exists())


if __name__ == "__main__":
    unittest.main()
