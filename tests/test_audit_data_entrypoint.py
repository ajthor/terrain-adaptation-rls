import contextlib
import io
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.experiments import audit_data


class AuditDataEntrypointTests(unittest.TestCase):
    def test_audit_data_main_prints_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "warty" / "scene0"
            scene_dir.mkdir(parents=True)
            (scene_dir / "odom.csv").write_text("time,x\n0.0,1\n")
            (scene_dir / "cmd_vel.csv").write_text("time,u\n0.0,0\n")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                exit_code = audit_data.main(["--data-root", tmpdir, "--platform", "warty"])

        self.assertEqual(exit_code, 0)
        self.assertIn("platform,scene,odom_rows", stdout.getvalue())
        self.assertIn("warty,scene0,1,1,0.0,False", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
