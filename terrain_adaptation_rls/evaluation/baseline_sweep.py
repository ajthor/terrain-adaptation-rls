"""Aggregate online baseline comparisons across scenes and windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping

from terrain_adaptation_rls.configuration import ExperimentConfig, load_config
from terrain_adaptation_rls.evaluation.artifacts import write_json


@dataclass(frozen=True)
class SweepWindow:
    """One scene/window slice used in aggregate evaluation."""

    scene: str
    split: str
    start_index: int
    window_index: int


@dataclass(frozen=True)
class SweepArtifacts:
    """Outputs from a completed baseline sweep."""

    artifact_dir: Path
    rows: list[dict[str, object]]
    method_summary: list[dict[str, object]]


def run_baseline_sweep(
    *,
    fe_run_dir: str | Path,
    artifact_dir: str | Path,
    neuralfly_run_dir: str | Path | None = None,
    node_run_dir: str | Path | None = None,
    scenes: list[str] | None = None,
    split: str = "heldout",
    device: str = "cpu",
    max_points: int = 512,
    window_stride: int | None = None,
    max_windows_per_scene: int = 1,
    n_example_points: int | None = None,
    forgetting_factor: float = 0.95,
    initial_covariance: float = 1_000.0,
    measurement_noise: float = 1e-6,
    include_fe_variants: bool = True,
    kalman_process_noise: float = 0.0,
    fe_sgd_learning_rate: float = 1.0,
    fe_sgd_momentum: float = 0.0,
    fe_sgd_weight_decay: float = 0.0,
    fe_window_size: int = 100,
    fe_window_ridge: float = 1e-6,
    linear_include_bias: bool = True,
    progress: bool = False,
) -> SweepArtifacts:
    """Run online baseline comparisons for multiple held-out scene windows."""

    from terrain_adaptation_rls.evaluation.online_baselines import (
        run_online_baseline_comparison,
    )

    fe_run_dir = Path(fe_run_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(fe_run_dir / "resolved_config.json")
    windows = resolve_sweep_windows(
        config,
        scenes=scenes,
        split=split,
        max_points=max_points,
        window_stride=window_stride,
        max_windows_per_scene=max_windows_per_scene,
    )
    rows: list[dict[str, object]] = []

    for window_number, window in enumerate(windows, start=1):
        if progress:
            print(
                f"[{window_number}/{len(windows)}] "
                f"evaluating {window.split}:{window.scene}@{window.start_index}",
                flush=True,
            )
        window_dir = artifact_dir / f"{window.split}_{window.scene}_start{window.start_index}"
        result = run_online_baseline_comparison(
            fe_run_dir=fe_run_dir,
            neuralfly_run_dir=neuralfly_run_dir,
            node_run_dir=node_run_dir,
            artifact_dir=window_dir,
            scene=window.scene,
            device=device,
            max_points=max_points,
            start_index=window.start_index,
            n_example_points=n_example_points,
            forgetting_factor=forgetting_factor,
            initial_covariance=initial_covariance,
            measurement_noise=measurement_noise,
            include_fe_variants=include_fe_variants,
            kalman_process_noise=kalman_process_noise,
            fe_sgd_learning_rate=fe_sgd_learning_rate,
            fe_sgd_momentum=fe_sgd_momentum,
            fe_sgd_weight_decay=fe_sgd_weight_decay,
            fe_window_size=fe_window_size,
            fe_window_ridge=fe_window_ridge,
            linear_include_bias=linear_include_bias,
        )
        if progress:
            methods = result.summary["methods"]
            best_method, best_metrics = min(
                methods.items(),
                key=lambda item: float(item[1]["mean_error"]),
            )
            print(
                f"[{window_number}/{len(windows)}] "
                f"best {best_method} mean_error={best_metrics['mean_error']:.6f}",
                flush=True,
            )
        rows.extend(summary_rows(result.summary, window))

    method_summary = summarize_method_rows(rows)
    write_sweep_artifacts(
        artifact_dir,
        windows=windows,
        rows=rows,
        method_summary=method_summary,
        command_summary={
            "fe_run_dir": fe_run_dir,
            "neuralfly_run_dir": neuralfly_run_dir,
            "node_run_dir": node_run_dir,
            "split": split,
            "scenes": scenes,
            "device": device,
            "max_points": max_points,
            "window_stride": window_stride,
            "max_windows_per_scene": max_windows_per_scene,
            "n_example_points": n_example_points,
            "forgetting_factor": forgetting_factor,
            "initial_covariance": initial_covariance,
            "measurement_noise": measurement_noise,
            "include_fe_variants": include_fe_variants,
            "kalman_process_noise": kalman_process_noise,
            "fe_sgd_learning_rate": fe_sgd_learning_rate,
            "fe_sgd_momentum": fe_sgd_momentum,
            "fe_sgd_weight_decay": fe_sgd_weight_decay,
            "fe_window_size": fe_window_size,
            "fe_window_ridge": fe_window_ridge,
            "linear_include_bias": linear_include_bias,
            "progress": progress,
        },
    )
    return SweepArtifacts(
        artifact_dir=artifact_dir,
        rows=rows,
        method_summary=method_summary,
    )


def resolve_sweep_windows(
    config: ExperimentConfig,
    *,
    scenes: list[str] | None,
    split: str,
    max_points: int,
    window_stride: int | None,
    max_windows_per_scene: int,
) -> list[SweepWindow]:
    """Resolve scene/window slices from config split metadata."""

    if config.platform is None:
        raise ValueError("Baseline sweep requires config.platform")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if max_windows_per_scene <= 0:
        raise ValueError("max_windows_per_scene must be positive")

    split_scenes = resolve_scene_splits(config, scenes=scenes, split=split)
    stride = max_points if window_stride is None else window_stride
    if stride <= 0:
        raise ValueError("window_stride must be positive")

    from terrain_adaptation_rls.data.load_data import load_scenes

    all_scenes = [scene for scene_names in split_scenes.values() for scene in scene_names]
    loaded = load_scenes(all_scenes, config.platform)
    windows: list[SweepWindow] = []
    for split_name, scene_names in split_scenes.items():
        for scene in scene_names:
            n_points = loaded[scene][0].shape[0]
            starts = window_starts(
                n_points=n_points,
                max_points=max_points,
                stride=stride,
                max_windows=max_windows_per_scene,
            )
            for window_index, start_index in enumerate(starts):
                windows.append(
                    SweepWindow(
                        scene=scene,
                        split=split_name,
                        start_index=start_index,
                        window_index=window_index,
                    )
                )
    if not windows:
        raise ValueError("baseline sweep resolved no scene windows")
    return windows


def resolve_scene_splits(
    config: ExperimentConfig,
    *,
    scenes: list[str] | None,
    split: str,
) -> dict[str, list[str]]:
    """Resolve scenes grouped by split name."""

    if scenes:
        return {"explicit": [str(scene) for scene in scenes]}

    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    test_scenes = [str(scene) for scene in config.data.get("test_scenes", ())]
    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]

    if split == "validation":
        return {"validation": validation_scenes}
    if split == "test":
        return {"test": test_scenes}
    if split == "heldout":
        result: dict[str, list[str]] = {}
        if validation_scenes:
            result["validation"] = validation_scenes
        if test_scenes:
            result["test"] = test_scenes
        return result
    if split == "train":
        return {"train": train_scenes}
    if split == "all_config":
        result = {}
        if train_scenes:
            result["train"] = train_scenes
        if validation_scenes:
            result["validation"] = validation_scenes
        if test_scenes:
            result["test"] = test_scenes
        return result

    raise ValueError(
        f"unknown split '{split}'; expected validation, test, heldout, train, or all_config"
    )


def window_starts(
    *,
    n_points: int,
    max_points: int,
    stride: int,
    max_windows: int,
) -> list[int]:
    """Return deterministic window starts that cover a scene without overlap assumptions."""

    if n_points <= 0:
        return []
    if n_points <= max_points:
        return [0]

    starts = list(range(0, n_points, stride))
    starts = [start for start in starts if start < n_points]
    starts = starts[:max_windows]
    if not starts:
        starts = [0]
    return starts


def summary_rows(
    summary: Mapping[str, object],
    window: SweepWindow,
) -> list[dict[str, object]]:
    """Flatten one comparison summary into per-method aggregate rows."""

    methods = summary["methods"]
    rows: list[dict[str, object]] = []
    for method_name, metrics in methods.items():
        row = {
            "split": window.split,
            "scene": window.scene,
            "window_index": window.window_index,
            "start_index": window.start_index,
            "n_steps": int(summary["n_steps"]),
            "method": method_name,
            "label": metrics["label"],
            "mean_error": float(metrics["mean_error"]),
            "final_accumulated_error": float(metrics["final_accumulated_error"]),
            "mse": float(metrics["mse"]),
            "mean_error_to_zero_delta_ratio": float(
                metrics["mean_error_to_zero_delta_ratio"]
            ),
        }
        for key, value in metrics.items():
            if key == "label" or key in row:
                continue
            if isinstance(value, (int, float)):
                row[str(key)] = float(value)
        rows.append(row)
    return rows


def summarize_method_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate flattened method rows across scenes/windows."""

    rows = list(rows)
    by_method: dict[str, list[Mapping[str, object]]] = {}
    labels: dict[str, str] = {}
    ranks_by_method = rank_methods_by_window(rows)
    for row in rows:
        method = str(row["method"])
        by_method.setdefault(method, []).append(row)
        labels[method] = str(row["label"])

    zero_mean = mean(
        float(row["mean_error"])
        for row in rows
        if str(row["method"]) == "zero_delta"
    )
    summaries: list[dict[str, object]] = []
    for method, method_rows in by_method.items():
        mean_errors = [float(row["mean_error"]) for row in method_rows]
        accumulated_errors = [float(row["final_accumulated_error"]) for row in method_rows]
        mses = [float(row["mse"]) for row in method_rows]
        method_mean = mean(mean_errors)
        ranks = ranks_by_method.get(method, [])
        summary = {
            "method": method,
            "label": labels[method],
            "n_windows": len(method_rows),
            "win_count": sum(1 for rank in ranks if rank == 1),
            "mean_rank": mean(ranks) if ranks else 0.0,
            "mean_error_mean": method_mean,
            "mean_error_median": median(mean_errors),
            "mean_error_std": _std(mean_errors),
            "final_accumulated_error_mean": mean(accumulated_errors),
            "mse_mean": mean(mses),
            "mean_error_to_zero_delta_ratio": _safe_ratio(method_mean, zero_mean),
            "relative_improvement_vs_zero_delta": 1.0 - _safe_ratio(
                method_mean,
                zero_mean,
            ),
        }
        summary.update(_aggregate_extra_numeric_fields(method_rows, summary.keys()))
        summaries.append(summary)

    return sorted(summaries, key=lambda item: float(item["mean_error_mean"]))


def summarize_method_rows_by_group(
    rows: Iterable[Mapping[str, object]],
    *,
    group_fields: tuple[str, ...] = ("split", "scene"),
) -> list[dict[str, object]]:
    """Aggregate method summaries independently for split/scene groups."""

    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        group_summary = summarize_method_rows(group_rows)
        for method_row in group_summary:
            prefix = dict(zip(group_fields, key))
            summaries.append({**prefix, **method_row})
    return summaries


def summarize_reference_comparisons(
    rows: Iterable[Mapping[str, object]],
    *,
    reference_method: str = "fe_rls",
) -> list[dict[str, object]]:
    """Summarize paired head-to-head comparisons against one reference method."""

    rows = list(rows)
    labels = {str(row["method"]): str(row["label"]) for row in rows}
    grouped = group_rows_by_window(rows)
    comparisons: dict[str, dict[str, object]] = {}

    for window_rows in grouped.values():
        by_method = {str(row["method"]): row for row in window_rows}
        reference_row = by_method.get(reference_method)
        if reference_row is None:
            continue
        reference_error = float(reference_row["mean_error"])

        for method, row in by_method.items():
            if method == reference_method:
                continue
            comparison_error = float(row["mean_error"])
            comparison = comparisons.setdefault(
                method,
                {
                    "reference_errors": [],
                    "comparison_errors": [],
                    "comparison_minus_reference": [],
                    "reference_win_count": 0,
                    "comparison_win_count": 0,
                    "tie_count": 0,
                },
            )
            reference_errors = comparison["reference_errors"]
            comparison_errors = comparison["comparison_errors"]
            deltas = comparison["comparison_minus_reference"]
            assert isinstance(reference_errors, list)
            assert isinstance(comparison_errors, list)
            assert isinstance(deltas, list)
            reference_errors.append(reference_error)
            comparison_errors.append(comparison_error)
            deltas.append(comparison_error - reference_error)

            if reference_error < comparison_error:
                comparison["reference_win_count"] = (
                    int(comparison["reference_win_count"]) + 1
                )
            elif comparison_error < reference_error:
                comparison["comparison_win_count"] = (
                    int(comparison["comparison_win_count"]) + 1
                )
            else:
                comparison["tie_count"] = int(comparison["tie_count"]) + 1

    summaries: list[dict[str, object]] = []
    for method, comparison in comparisons.items():
        reference_errors = comparison["reference_errors"]
        comparison_errors = comparison["comparison_errors"]
        deltas = comparison["comparison_minus_reference"]
        assert isinstance(reference_errors, list)
        assert isinstance(comparison_errors, list)
        assert isinstance(deltas, list)
        reference_mean = mean(reference_errors)
        comparison_mean = mean(comparison_errors)
        summaries.append(
            {
                "reference_method": reference_method,
                "reference_label": labels.get(reference_method, reference_method),
                "comparison_method": method,
                "comparison_label": labels.get(method, method),
                "n_windows": len(deltas),
                "reference_win_count": int(comparison["reference_win_count"]),
                "comparison_win_count": int(comparison["comparison_win_count"]),
                "tie_count": int(comparison["tie_count"]),
                "reference_mean_error": reference_mean,
                "comparison_mean_error": comparison_mean,
                "comparison_minus_reference_mean_error": mean(deltas),
                "comparison_minus_reference_median_error": median(deltas),
                "comparison_minus_reference_std_error": _std(deltas),
                "reference_relative_improvement": 1.0
                - _safe_ratio(reference_mean, comparison_mean),
            }
        )

    return sorted(
        summaries,
        key=lambda item: float(item["comparison_mean_error"]),
    )


def rank_methods_by_window(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, list[int]]:
    """Rank methods independently within each scene/window."""

    grouped = group_rows_by_window(rows)
    ranks: dict[str, list[int]] = {}
    for window_rows in grouped.values():
        sorted_rows = sorted(window_rows, key=lambda row: float(row["mean_error"]))
        for rank, row in enumerate(sorted_rows, start=1):
            ranks.setdefault(str(row["method"]), []).append(rank)
    return ranks


def group_rows_by_window(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str, int, int], list[Mapping[str, object]]]:
    """Group flattened method rows by split, scene, start, and window index."""

    grouped: dict[tuple[str, str, int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("split", "")),
            str(row.get("scene", "")),
            int(row.get("start_index", 0)),
            int(row.get("window_index", 0)),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def write_sweep_artifacts(
    artifact_dir: Path,
    *,
    windows: list[SweepWindow],
    rows: list[dict[str, object]],
    method_summary: list[dict[str, object]],
    command_summary: Mapping[str, object],
) -> None:
    """Write CSV, JSON, and aggregate plots for a baseline sweep."""

    scene_summary = summarize_method_rows_by_group(rows)
    pairwise_summary = summarize_reference_comparisons(rows)
    write_json(
        artifact_dir / "summary.json",
        {
            "command": command_summary,
            "n_windows": len(windows),
            "windows": [window.__dict__ for window in windows],
            "method_summary": method_summary,
            "scene_method_summary": scene_summary,
            "pairwise_vs_fe_rls": pairwise_summary,
        },
    )
    write_rows_csv(artifact_dir / "window_metrics.csv", rows)
    write_rows_csv(artifact_dir / "method_summary.csv", method_summary)
    write_rows_csv(artifact_dir / "scene_method_summary.csv", scene_summary)
    write_rows_csv(artifact_dir / "pairwise_vs_fe_rls.csv", pairwise_summary)
    write_method_bar_plot(artifact_dir / "mean_error_by_method.png", method_summary)
    write_per_window_plot(artifact_dir / "mean_error_by_window.png", rows)


def write_rows_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    """Write a list of dictionaries as a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_method_bar_plot(path: Path, method_summary: list[Mapping[str, object]]) -> None:
    """Write aggregate mean-error bar plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in method_summary]
    values = [float(row["mean_error_mean"]) for row in method_summary]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean prediction error")
    ax.set_title("Aggregate held-out error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_per_window_plot(path: Path, rows: list[Mapping[str, object]]) -> None:
    """Write per-window mean-error lines for each method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_keys = []
    for row in rows:
        key = f"{row['split']}:{row['scene']}:{row['start_index']}"
        if key not in window_keys:
            window_keys.append(key)

    methods = []
    labels = {}
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
        labels[method] = str(row["label"])

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(window_keys)), 5))
    for method in methods:
        values = []
        for key in window_keys:
            value = next(
                (
                    float(row["mean_error"])
                    for row in rows
                    if str(row["method"]) == method
                    and f"{row['split']}:{row['scene']}:{row['start_index']}" == key
                ),
                None,
            )
            values.append(value)
        ax.plot(range(len(window_keys)), values, marker="o", linewidth=1.2, label=labels[method])
    ax.set_xticks(range(len(window_keys)))
    ax.set_xticklabels(window_keys, rotation=35, ha="right")
    ax.set_ylabel("mean prediction error")
    ax.set_title("Per-window held-out error")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _aggregate_extra_numeric_fields(
    rows: list[Mapping[str, object]],
    existing_summary_keys: Iterable[str],
) -> dict[str, float]:
    """Aggregate extra numeric per-window fields as ``<field>_mean`` columns."""

    existing = set(existing_summary_keys)
    skip_fields = {
        "split",
        "scene",
        "window_index",
        "start_index",
        "n_steps",
        "method",
        "label",
        "mean_error",
        "final_accumulated_error",
        "mse",
        "mean_error_to_zero_delta_ratio",
    }
    numeric_fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            target_key = _aggregate_field_name(str(key))
            if key in skip_fields or target_key in existing:
                continue
            if str(key).startswith("logged_k") and str(key).endswith("_n_windows"):
                continue
            if isinstance(value, (int, float)) and key not in numeric_fields:
                numeric_fields.append(str(key))

    summary: dict[str, float] = {}
    for key in numeric_fields:
        target_key = _aggregate_field_name(key)
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float))
        ]
        if values:
            summary[target_key] = mean(values)
    return summary


def _aggregate_field_name(field: str) -> str:
    if field.startswith("logged_k") and field.endswith("_mean"):
        return field
    return f"{field}_mean"


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    variance = sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
