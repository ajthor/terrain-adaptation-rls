import unittest

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.evaluation.baseline_sweep import (
    rank_methods_by_window,
    resolve_scene_splits,
    summarize_method_rows_by_group,
    summarize_method_rows,
    summarize_reference_comparisons,
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
            _row("zero_delta", "zero", 10.0, window_index=0),
            _row("method_a", "A", 2.0, window_index=0),
            _row("method_b", "B", 3.0, window_index=0),
            _row("zero_delta", "zero", 8.0, window_index=1),
            _row("method_a", "A", 4.0, window_index=1),
            _row("method_b", "B", 1.0, window_index=1),
        ]

        summary = summarize_method_rows(rows)

        self.assertEqual(summary[0]["method"], "method_b")
        self.assertEqual(summary[1]["method"], "method_a")
        self.assertEqual(summary[-1]["method"], "zero_delta")
        self.assertAlmostEqual(summary[0]["mean_error_mean"], 2.0)
        self.assertAlmostEqual(summary[0]["relative_improvement_vs_zero_delta"], 1 - 2 / 9)
        self.assertEqual(summary[0]["win_count"], 1)
        self.assertAlmostEqual(summary[0]["mean_rank"], 1.5)

    def test_summarize_method_rows_aggregates_extra_metrics_readably(self):
        rows = [
            _row("method_a", "A", 2.0, window_index=0),
            _row("zero_delta", "zero", 10.0, window_index=0),
            _row("method_a", "A", 4.0, window_index=1),
            _row("zero_delta", "zero", 8.0, window_index=1),
        ]
        rows[0]["p95_error"] = 3.0
        rows[1]["p95_error"] = 11.0
        rows[2]["p95_error"] = 5.0
        rows[3]["p95_error"] = 9.0
        rows[0]["logged_k10_endpoint_error_mean"] = 7.0
        rows[1]["logged_k10_endpoint_error_mean"] = 20.0
        rows[2]["logged_k10_endpoint_error_mean"] = 9.0
        rows[3]["logged_k10_endpoint_error_mean"] = 18.0
        rows[0]["logged_k10_n_windows"] = 42.0
        rows[0]["recursive_k10_final_step_error_mean"] = 11.0
        rows[1]["recursive_k10_final_step_error_mean"] = 30.0
        rows[2]["recursive_k10_final_step_error_mean"] = 13.0
        rows[3]["recursive_k10_final_step_error_mean"] = 28.0
        rows[0]["recursive_k10_n_rollouts"] = 8.0

        summary = summarize_method_rows(rows)
        method_a = summary[0]

        self.assertAlmostEqual(method_a["p95_error_mean"], 4.0)
        self.assertAlmostEqual(method_a["logged_k10_endpoint_error_mean"], 8.0)
        self.assertAlmostEqual(method_a["recursive_k10_final_step_error_mean"], 12.0)
        self.assertNotIn("logged_k10_endpoint_error_mean_mean", method_a)
        self.assertNotIn("logged_k10_n_windows_mean", method_a)
        self.assertNotIn("recursive_k10_final_step_error_mean_mean", method_a)
        self.assertNotIn("recursive_k10_n_rollouts_mean", method_a)

    def test_rank_methods_by_window(self):
        rows = [
            _row("method_a", "A", 1.0, window_index=0),
            _row("method_b", "B", 2.0, window_index=0),
            _row("method_a", "A", 3.0, window_index=1),
            _row("method_b", "B", 1.5, window_index=1),
        ]

        ranks = rank_methods_by_window(rows)

        self.assertEqual(ranks["method_a"], [1, 2])
        self.assertEqual(ranks["method_b"], [2, 1])

    def test_summarize_method_rows_by_group(self):
        rows = [
            _row("method_a", "A", 1.0, scene="scene0", window_index=0),
            _row("zero_delta", "zero", 10.0, scene="scene0", window_index=0),
            _row("method_a", "A", 3.0, scene="scene1", window_index=0),
            _row("zero_delta", "zero", 6.0, scene="scene1", window_index=0),
        ]

        summary = summarize_method_rows_by_group(rows)

        self.assertEqual(summary[0]["split"], "test")
        self.assertEqual(summary[0]["scene"], "scene0")
        self.assertEqual(summary[0]["method"], "method_a")
        self.assertAlmostEqual(summary[0]["relative_improvement_vs_zero_delta"], 0.9)
        self.assertEqual(summary[2]["scene"], "scene1")
        self.assertEqual(summary[2]["method"], "method_a")

    def test_summarize_reference_comparisons(self):
        rows = [
            _row("fe_rls", "FE-RLS", 1.0, window_index=0),
            _row("method_a", "A", 2.0, window_index=0),
            _row("method_b", "B", 0.5, window_index=0),
            _row("fe_rls", "FE-RLS", 2.0, window_index=1),
            _row("method_a", "A", 3.0, window_index=1),
            _row("method_b", "B", 2.5, window_index=1),
        ]

        summary = summarize_reference_comparisons(rows, reference_method="fe_rls")
        by_method = {row["comparison_method"]: row for row in summary}

        self.assertEqual(by_method["method_a"]["reference_win_count"], 2)
        self.assertEqual(by_method["method_a"]["comparison_win_count"], 0)
        self.assertAlmostEqual(
            by_method["method_a"]["comparison_minus_reference_mean_error"],
            1.0,
        )
        self.assertAlmostEqual(
            by_method["method_a"]["reference_relative_improvement"],
            1.0 - 1.5 / 2.5,
        )
        self.assertEqual(by_method["method_b"]["comparison_win_count"], 1)


def _row(method, label, mean_error, *, window_index, scene="scene0"):
    return {
        "split": "test",
        "scene": scene,
        "window_index": window_index,
        "start_index": 512 * window_index,
        "method": method,
        "label": label,
        "mean_error": mean_error,
        "final_accumulated_error": 2.0 * mean_error,
        "mse": 0.5 * mean_error,
    }


if __name__ == "__main__":
    unittest.main()
