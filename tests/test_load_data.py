import tempfile
from pathlib import Path
import unittest

import numpy as np

try:
    from terrain_adaptation_rls.data.load_data import load_csv, scene_csv_paths
except ModuleNotFoundError:
    load_csv = None
    scene_csv_paths = None


@unittest.skipIf(load_csv is None, "torch is not installed")
class LoadDataTests(unittest.TestCase):
    def test_load_csv_uses_header_and_keeps_2d_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            path.write_text("a,b\n1.0,2.0\n")

            data = load_csv(path.as_posix())

        self.assertEqual(data.shape, (1, 2))
        self.assertTrue(np.allclose(data, np.array([[1.0, 2.0]])))

    def test_small_ugv_scene_paths_are_namespaced(self):
        odom_path, cmd_vel_path = scene_csv_paths("small_ugv", "jackal_0770/grass")

        self.assertEqual(odom_path, "terrain_adaptation_rls/data/jackal_0770/grass/odom.csv")
        self.assertEqual(
            cmd_vel_path,
            "terrain_adaptation_rls/data/jackal_0770/grass/cmd_vel.csv",
        )

    def test_small_ugv_scene_paths_reject_unnamespaced_scene(self):
        with self.assertRaises(ValueError):
            scene_csv_paths("small_ugv", "grass")


if __name__ == "__main__":
    unittest.main()
