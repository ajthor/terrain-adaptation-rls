import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.experiments import train
from terrain_adaptation_rls.experiments import eval_fe_rls
from terrain_adaptation_rls.experiments import train_fe
from terrain_adaptation_rls.training import DistributedContext

try:
    import torch
except ModuleNotFoundError:
    torch = None


class TrainingEntrypointTests(unittest.TestCase):
    def test_distributed_context_defaults_to_single_process(self):
        context = DistributedContext.from_env({})

        self.assertEqual(context.rank, 0)
        self.assertEqual(context.local_rank, 0)
        self.assertEqual(context.world_size, 1)
        self.assertFalse(context.is_distributed)
        self.assertTrue(context.is_rank_zero)

    def test_distributed_context_reads_torchrun_environment(self):
        context = DistributedContext.from_env(
            {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
        )

        self.assertEqual(context.rank, 2)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.world_size, 4)
        self.assertTrue(context.is_distributed)
        self.assertFalse(context.is_rank_zero)

    def test_train_main_creates_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug_train",
                        "kind": "train",
                        "output_root": (Path(tmpdir) / "outputs").as_posix(),
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = train.main(["--config", config_path.as_posix()])

            run_dirs = list((Path(tmpdir) / "outputs" / "train").iterdir())
            distributed = json.loads((run_dirs[0] / "distributed.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(run_dirs), 1)
        self.assertEqual(distributed["world_size"], 1)

    def test_train_dry_run_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug_train",
                        "kind": "train",
                        "methods": ["node_static"],
                        "output_root": (Path(tmpdir) / "outputs").as_posix(),
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = train.main(["--config", config_path.as_posix(), "--dry-run"])

            output_dir = Path(tmpdir) / "outputs"

        self.assertEqual(exit_code, 0)
        self.assertIn("valid train config", stdout.getvalue())
        self.assertFalse(output_dir.exists())

    @unittest.skipIf(torch is None, "torch/function_encoder is not installed")
    def test_train_main_smoke_train_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug_train",
                        "kind": "train",
                        "output_root": (Path(tmpdir) / "outputs").as_posix(),
                        "model": {
                            "family": "neural_ode",
                            "hidden_size": 8,
                            "n_basis": 2,
                        },
                        "training": {
                            "steps": 1,
                            "batch_size": 2,
                            "n_points": 3,
                        },
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = train.main(
                    [
                        "--config",
                        config_path.as_posix(),
                        "--smoke-train",
                        "--device",
                        "cpu",
                    ]
                )

            run_dirs = list((Path(tmpdir) / "outputs" / "train").iterdir())
            metrics = json.loads((run_dirs[0] / "training_smoke.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(metrics["steps"], 1)

    def test_train_fe_dry_run_validates_fe_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_fe.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug_fe",
                        "kind": "train",
                        "platform": "warty",
                        "output_root": (Path(tmpdir) / "outputs").as_posix(),
                        "model": {
                            "family": "function_encoder",
                            "hidden_size": 8,
                            "n_basis": 2,
                        },
                        "training": {
                            "steps": 1,
                        },
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = train_fe.main(["--config", config_path.as_posix(), "--dry-run"])

            output_dir = Path(tmpdir) / "outputs"

        self.assertEqual(exit_code, 0)
        self.assertIn("valid FE train config", stdout.getvalue())
        self.assertFalse(output_dir.exists())

    def test_eval_fe_rls_dry_run_validates_train_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_run_dir = Path(tmpdir) / "train_run"
            train_run_dir.mkdir()
            (train_run_dir / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "name": "debug_fe",
                        "kind": "train",
                        "platform": "warty",
                        "model": {"family": "function_encoder"},
                    }
                )
            )
            (train_run_dir / "function_encoder_model.pth").write_bytes(b"placeholder")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = eval_fe_rls.main(
                    ["--train-run-dir", train_run_dir.as_posix(), "--dry-run"]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("valid FE-RLS eval", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
