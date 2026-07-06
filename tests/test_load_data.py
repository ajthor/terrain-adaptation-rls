import tempfile
from pathlib import Path
import unittest

import numpy as np

try:
    from terrain_adaptation_rls.data.load_data import load_csv
except ModuleNotFoundError:
    load_csv = None


@unittest.skipIf(load_csv is None, "torch is not installed")
class LoadDataTests(unittest.TestCase):
    def test_load_csv_uses_header_and_keeps_2d_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            path.write_text("a,b\n1.0,2.0\n")

            data = load_csv(path.as_posix())

        self.assertEqual(data.shape, (1, 2))
        self.assertTrue(np.allclose(data, np.array([[1.0, 2.0]])))


if __name__ == "__main__":
    unittest.main()
