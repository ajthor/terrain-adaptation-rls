from pathlib import Path
import unittest

from terrain_adaptation_rls.configuration import load_config
from terrain_adaptation_rls.methods import validate_method_names


class ExampleConfigTests(unittest.TestCase):
    def test_all_example_configs_load(self):
        config_paths = sorted(Path("configs").rglob("*.json"))

        self.assertGreater(len(config_paths), 0)
        for path in config_paths:
            with self.subTest(path=path):
                config = load_config(path)
                validate_method_names(config.methods)


if __name__ == "__main__":
    unittest.main()
