import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config, write_config


class ConfigurationTests(unittest.TestCase):
    def test_from_mapping_keeps_known_fields_and_extras(self):
        config = ExperimentConfig.from_mapping(
            {
                "name": "debug",
                "kind": "eval",
                "seed": "3",
                "platform": "warty",
                "methods": "fe_rls",
                "data": {"split": "scene_holdout"},
                "custom": {"x": 1},
            }
        )

        self.assertEqual(config.name, "debug")
        self.assertEqual(config.kind, "eval")
        self.assertEqual(config.seed, 3)
        self.assertEqual(config.methods, ("fe_rls",))
        self.assertEqual(config.extras, {"custom": {"x": 1}})
        self.assertEqual(config.to_dict()["custom"], {"x": 1})

    def test_from_mapping_requires_name_and_kind(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_mapping({"kind": "train"})
        with self.assertRaises(ValueError):
            ExperimentConfig.from_mapping({"name": "missing_kind"})

    def test_load_and_write_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "config.json"
            source.write_text(
                json.dumps(
                    {
                        "name": "debug_train",
                        "kind": "train",
                        "methods": ["node_static"],
                        "training": {"steps": 2},
                    }
                )
            )

            config = load_config(source)
            target = Path(tmpdir) / "resolved.json"
            write_config(target, config)
            resolved = json.loads(target.read_text())

        self.assertEqual(config.methods, ("node_static",))
        self.assertEqual(resolved["training"], {"steps": 2})


if __name__ == "__main__":
    unittest.main()
