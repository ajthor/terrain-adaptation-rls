import unittest

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.evaluation.baseline_sweep import (
    resolve_scene_splits,
    summarize_method_rows,
    window_starts,
)


class BaselineSweepTests(unittest.TestCase):
    def test_resolve_scene_splits_uses_heldout_config_scenes(self):
        config = ExperimentConfig(
            name="debug",
            kind="train",
            platform="warty",
            data={
                "train_scenes": ["scene0"],
                "validation_scenes": ["scene1"],
                "test_scenes": ["scene5"],
            },
        )

        splits = resolve_scene_splits(config, scenes=None, split="heldout")

        self.assertEqual(splits, {"validation": ["scene1"], "test": ["scene5"]})

    def test_explicit_scenes_override_split(self):
        config = ExperimentConfig(name="debug", kind="train", platform="warty")

        splits = resolve_scene_splits(config, scenes=["scene2", "scene7"], split="heldout")

        self.assertEqual(splits, {"explicit": ["scene2", "scene7"]})

    def test_window_starts_respects_max_windows(self):
        starts = window_starts(
            n_points=1300,
            max_points=512,
            stride=256,
            max_windows=3,
        )

        self.assertEqual(starts, [0, 256, 512])

    def test_summarize_method_rows_sorts_by_mean_error(self):
        rows = [
            {"method": "zero_delta", "label": "zero", "mean_error": 10.0, "final_accumulated_error": 20.0, "mse": 5.0},
            {"method": "method_a", "label": "A", "mean_error": 2.0, "final_accumulated_error": 4.0, "mse": 1.0},
            {"method": "method_b", "label": "B", "mean_error": 3.0, "final_accumulated_error": 6.0, "mse": 2.0},
            {"method": "zero_delta", "label": "zero", "mean_error": 8.0, "final_accumulated_error": 16.0, "mse": 4.0},
            {"method": "method_a", "label": "A", "mean_error": 4.0, "final_accumulated_error": 8.0, "mse": 2.0},
            {"method": "method_b", "label": "B", "mean_error": 1.0, "final_accumulated_error": 2.0, "mse": 1.0},
        ]

        summary = summarize_method_rows(rows)

        self.assertEqual(summary[0]["method"], "method_b")
        self.assertEqual(summary[1]["method"], "method_a")
        self.assertEqual(summary[-1]["method"], "zero_delta")
        self.assertAlmostEqual(summary[0]["mean_error_mean"], 2.0)
        self.assertAlmostEqual(summary[0]["relative_improvement_vs_zero_delta"], 1 - 2 / 9)


if __name__ == "__main__":
    unittest.main()
