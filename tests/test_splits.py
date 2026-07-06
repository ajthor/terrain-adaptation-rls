import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.data.splits import (
    SceneSplitPolicy,
    scene_source_metadata,
    split_indices,
)


class SplitPolicyTests(unittest.TestCase):
    def test_split_indices_is_deterministic(self):
        first = split_indices(20, seed=3)
        second = split_indices(20, seed=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 16)
        self.assertEqual(len(first["validation"]), 2)
        self.assertEqual(len(first["test"]), 2)
        self.assertEqual(
            sorted(first["train"] + first["validation"] + first["test"]),
            list(range(20)),
        )

    def test_split_indices_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            split_indices(10, seed=0, train_fraction=0.9, validation_fraction=0.2)

    def test_scene_split_manifest_writes_json(self):
        policy = SceneSplitPolicy(
            name="warty_debug",
            platform="warty",
            seed=5,
            train_scenes=("scene0", "scene2"),
            validation_scenes=("scene5",),
            test_scenes=("scene1",),
        )
        manifest = policy.manifest(
            preprocessing_version="test",
            source_metadata=scene_source_metadata({"scene0": 10, "scene1": 7}),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.write_json(path)
            data = json.loads(path.read_text())

        self.assertEqual(data["name"], "warty_debug")
        self.assertEqual(data["platform"], "warty")
        self.assertEqual(data["train_scenes"], ["scene0", "scene2"])
        self.assertEqual(data["source_metadata"]["scene0"]["rows"], 10)


if __name__ == "__main__":
    unittest.main()
