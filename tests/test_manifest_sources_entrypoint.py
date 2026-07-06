import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.experiments import manifest_sources


class ManifestSourcesEntrypointTests(unittest.TestCase):
    def test_manifest_sources_main(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "warty" / "scene0"
            scene_dir.mkdir(parents=True)
            (scene_dir / "odom.csv").write_text("time,x\n0.0,1\n")
            (scene_dir / "cmd_vel.csv").write_text("time,u\n0.0,0\n")
            output = Path(tmpdir) / "manifest.json"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = manifest_sources.main(
                    [
                        "--data-root",
                        tmpdir,
                        "--platform",
                        "warty",
                        "--output",
                        output.as_posix(),
                    ]
                )

            manifest = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["metadata"]["platform"], "warty")
        self.assertEqual(manifest["scenes"][0]["scene"], "scene0")


if __name__ == "__main__":
    unittest.main()
