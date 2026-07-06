import json
import contextlib
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import unittest

from terrain_adaptation_rls.experiments.common import prepare_run
from terrain_adaptation_rls.experiments import eval_k_step, eval_streaming


class ExperimentEntrypointTests(unittest.TestCase):
    def test_prepare_run_writes_resolved_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            output_root = Path(tmpdir) / "outputs"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug",
                        "kind": "eval",
                        "platform": "warty",
                        "methods": ["fe_rls"],
                        "output_root": output_root.as_posix(),
                    }
                )
            )

            prepared = prepare_run(
                config_path,
                command="eval_streaming",
                timestamp=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
            )

            resolved = json.loads((prepared.run_dir / "resolved_config.json").read_text())
            summary = json.loads((prepared.run_dir / "summary.json").read_text())

        self.assertEqual(prepared.run_dir.name, "20260706T120000Z_debug")
        self.assertEqual(resolved["methods"], ["fe_rls"])
        self.assertEqual(summary["command"], "eval_streaming")
        self.assertEqual(summary["method_specs"][0]["name"], "fe_rls")
        self.assertEqual(summary["status"], "prepared")

    def test_prepare_run_rejects_unknown_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "debug",
                        "kind": "eval",
                        "methods": ["missing_method"],
                    }
                )
            )

            with self.assertRaisesRegex(KeyError, "Known methods"):
                prepare_run(config_path, command="eval_streaming")

    def test_eval_streaming_main_creates_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, name="streaming")

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = eval_streaming.main(["--config", config_path.as_posix()])

            run_dirs = list((Path(tmpdir) / "outputs" / "eval").iterdir())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(run_dirs), 1)

    def test_eval_streaming_dry_run_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, name="streaming")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = eval_streaming.main(
                    ["--config", config_path.as_posix(), "--dry-run"]
                )

            output_dir = Path(tmpdir) / "outputs"
        self.assertEqual(exit_code, 0)
        self.assertIn("valid eval config", stdout.getvalue())
        self.assertFalse(output_dir.exists())

    def test_eval_k_step_main_creates_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config(tmpdir, name="k_step")

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = eval_k_step.main(["--config", config_path.as_posix()])

            run_dirs = list((Path(tmpdir) / "outputs" / "eval").iterdir())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(run_dirs), 1)


def _write_config(tmpdir: str, *, name: str) -> Path:
    config_path = Path(tmpdir) / f"{name}.json"
    config_path.write_text(
        json.dumps(
            {
                "name": name,
                "kind": "eval",
                "output_root": (Path(tmpdir) / "outputs").as_posix(),
            }
        )
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
