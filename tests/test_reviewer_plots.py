from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib  # noqa: F401

    from terrain_adaptation_rls.evaluation.reviewer_plots import (
        write_reviewer_comparison_plots,
    )
except ModuleNotFoundError:
    matplotlib = None


@unittest.skipIf(matplotlib is None, "matplotlib is not installed")
class ReviewerPlotTests(unittest.TestCase):
    def test_writes_reviewer_comparison_plot_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene1 = root / "scene1"
            scene5 = root / "scene5"
            scene1.mkdir()
            scene5.mkdir()
            _write_sweep_csvs(scene1, "scene1")
            _write_sweep_csvs(scene5, "scene5")

            payload = write_reviewer_comparison_plots(
                run_dirs={"scene1": scene1, "scene5": scene5},
                artifact_dir=root / "plots",
            )

            plot_dir = root / "plots"
            self.assertIn("representative_mean_error.png", payload["plots"])
            self.assertFalse(payload["include_zero_delta"])
            self.assertNotIn("zero_delta", payload["representative_methods"])
            self.assertTrue((plot_dir / "all_methods_ranked_mean_error.png").is_file())
            self.assertTrue((plot_dir / "fe_variant_mean_error.png").is_file())
            self.assertTrue((plot_dir / "logged_k_step_endpoint_error.png").is_file())
            self.assertTrue((plot_dir / "logged_k_step_accumulated_error.png").is_file())
            self.assertTrue((plot_dir / "logged_k_step_trajectory_rmse.png").is_file())
            self.assertTrue(
                (plot_dir / "logged_k_step_integral_square_error.png").is_file()
            )
            self.assertTrue((plot_dir / "recursive_k_step_final_error.png").is_file())
            self.assertTrue(
                (plot_dir / "recursive_k_step_accumulated_error.png").is_file()
            )
            self.assertTrue((plot_dir / "recursive_k_step_trajectory_rmse.png").is_file())
            self.assertTrue(
                (plot_dir / "recursive_k_step_integral_square_error.png").is_file()
            )
            self.assertTrue((plot_dir / "plot_summary.csv").is_file())


def _write_sweep_csvs(path: Path, scene: str) -> None:
    method_rows = [
        _method_row("offline_fe", "offline FE", 0.04),
        _method_row("fe_rls", "FE-RLS", 0.03),
        _method_row("fe_kalman", "FE-Kalman", 0.035),
        _method_row("fe_window_ls", "FE-window LS", 0.036),
        _method_row("fe_sgd", "FE-SGD", 0.08),
        _method_row("neuralfly_rls", "NeuralFly-style RLS", 0.05),
        _method_row("maml_online", "MAML-online", 0.06),
        _method_row("static_node", "static NODE", 0.07),
        _method_row("linear_rls", "linear RLS", 0.09),
        _method_row("zero_delta", "zero delta", 0.3),
    ]
    _write_csv(path / "method_summary.csv", method_rows)

    window_rows = []
    for window_index in range(2):
        for row in method_rows:
            window_rows.append(
                {
                    "split": "explicit",
                    "scene": scene,
                    "window_index": window_index,
                    "start_index": 512 * window_index,
                    "method": row["method"],
                    "label": row["label"],
                    "mean_error": float(row["mean_error_mean"]) + 0.001 * window_index,
                }
            )
    _write_csv(path / "window_metrics.csv", window_rows)


def _method_row(method: str, label: str, mean_error: float) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method,
        "label": label,
        "n_windows": 2,
        "win_count": 0,
        "mean_rank": 1,
        "mean_error_mean": mean_error,
        "mean_error_median": mean_error,
        "mean_error_std": 0.0,
        "integrated_position_mean_error_mean": 10.0 * mean_error,
    }
    for horizon in (1, 5, 10, 20, 50):
        row[f"logged_k{horizon}_endpoint_error_mean"] = mean_error * horizon
        row[f"logged_k{horizon}_accumulated_error_mean"] = mean_error * horizon * 2
        row[f"logged_k{horizon}_trajectory_rmse_mean"] = mean_error * horizon**0.5
        row[f"logged_k{horizon}_integral_square_error_mean"] = mean_error**2 * horizon
        row[f"recursive_k{horizon}_final_step_error_mean"] = mean_error * horizon * 1.5
        row[f"recursive_k{horizon}_accumulated_error_mean"] = mean_error * horizon * 3
        row[f"recursive_k{horizon}_trajectory_rmse_mean"] = mean_error * horizon**0.5
        row[f"recursive_k{horizon}_integral_square_error_mean"] = mean_error**2 * horizon
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
