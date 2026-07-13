"""Reviewer-facing plots built from aggregate baseline sweep artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


REPRESENTATIVE_METHODS = (
    "offline_fe",
    "fe_rls",
    "fe_prior_rls",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "alpaca_online",
    "alpaca_prior_static",
    "maml_static",
    "maml_online",
    "static_node",
    "linear_rls",
)
FE_VARIANT_METHODS = (
    "offline_fe",
    "fe_rls",
    "fe_prior_static",
    "fe_prior_rls",
    "fe_kalman",
    "fe_window_ls",
    "fe_sgd",
)
K_STEP_METHODS = (
    "offline_fe",
    "fe_rls",
    "fe_prior_rls",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "alpaca_online",
    "maml_online",
    "static_node",
)
METHOD_COLORS = {
    "offline_fe": "#4b5563",
    "fe_rls": "#0f766e",
    "fe_prior_static": "#14b8a6",
    "fe_prior_rls": "#115e59",
    "fe_kalman": "#0891b2",
    "fe_window_ls": "#2563eb",
    "fe_sgd": "#7c3aed",
    "neuralfly_rls": "#d97706",
    "neuralfly_prior_static": "#f97316",
    "neuralfly_prior_rls": "#b45309",
    "offline_neuralfly": "#f59e0b",
    "alpaca_online": "#16a34a",
    "alpaca_prior_static": "#86efac",
    "offline_alpaca": "#22c55e",
    "maml_static": "#be185d",
    "maml_online": "#db2777",
    "static_node": "#dc2626",
    "linear_rls": "#6b7280",
    "offline_linear": "#9ca3af",
    "zero_delta": "#111827",
}
METHOD_STYLES = {
    "offline_fe": "-.",
    "fe_rls": "-",
    "fe_prior_static": ":",
    "fe_prior_rls": "-.",
    "neuralfly_rls": "--",
    "neuralfly_prior_static": ":",
    "neuralfly_prior_rls": "-.",
    "alpaca_online": "-",
    "alpaca_prior_static": ":",
    "offline_alpaca": "-.",
    "maml_static": "--",
    "maml_online": ":",
    "static_node": "-.",
    "linear_rls": "--",
    "zero_delta": "--",
}


def write_reviewer_comparison_plots(
    *,
    run_dirs: Mapping[str, str | Path],
    artifact_dir: str | Path,
    include_zero_delta: bool = False,
) -> dict[str, object]:
    """Write grouped comparison plots from one or more baseline sweep runs."""

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    method_summaries = {
        scene: load_csv_rows(Path(run_dir) / "method_summary.csv")
        for scene, run_dir in run_dirs.items()
    }
    window_metrics = {
        scene: load_csv_rows(Path(run_dir) / "window_metrics.csv")
        for scene, run_dir in run_dirs.items()
    }
    representative_methods = _maybe_with_zero(REPRESENTATIVE_METHODS, include_zero_delta)
    k_step_methods = _maybe_with_zero(K_STEP_METHODS, include_zero_delta)

    written = [
        write_all_methods_ranked_plot(
            artifact_path / "all_methods_ranked_mean_error.png",
            method_summaries,
        ),
        write_grouped_method_bar_plot(
            artifact_path / "representative_mean_error.png",
            method_summaries,
            methods=representative_methods,
            title="Representative baseline one-step error",
            ylabel="mean one-step error",
            log_scale=True,
        ),
        write_grouped_method_bar_plot(
            artifact_path / "fe_variant_mean_error.png",
            method_summaries,
            methods=FE_VARIANT_METHODS,
            title="Function Encoder update variants",
            ylabel="mean one-step error",
            log_scale=False,
        ),
        write_per_window_plot(
            artifact_path / "representative_per_window_error.png",
            window_metrics,
            methods=representative_methods,
        ),
        write_grouped_method_bar_plot(
            artifact_path / "adaptation_samples_25pct.png",
            method_summaries,
            methods=representative_methods,
            title="Samples to 25% online error reduction",
            ylabel="samples",
            metric="adaptation_samples_to_25pct_improvement_mean",
            log_scale=False,
        ),
        write_k_step_plot(
            artifact_path / "logged_k_step_endpoint_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="logged",
            value_name="endpoint_error",
            title="Logged-input k-step endpoint error",
            ylabel="endpoint error",
        ),
        write_k_step_plot(
            artifact_path / "logged_k_step_accumulated_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="logged",
            value_name="accumulated_error",
            title="Logged-input k-step accumulated error",
            ylabel="accumulated error",
        ),
        write_k_step_plot(
            artifact_path / "logged_k_step_trajectory_rmse.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="logged",
            value_name="trajectory_rmse",
            title="Logged-input k-step trajectory RMSE",
            ylabel="trajectory RMSE",
        ),
        write_k_step_plot(
            artifact_path / "logged_k_step_integral_square_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="logged",
            value_name="integral_square_error",
            title="Logged-input k-step integral square error",
            ylabel="integral square error",
        ),
        write_k_step_plot(
            artifact_path / "recursive_k_step_final_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="recursive",
            value_name="final_step_error",
            title="Recursive open-loop k-step final error",
            ylabel="final step error",
        ),
        write_k_step_plot(
            artifact_path / "recursive_k_step_accumulated_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="recursive",
            value_name="accumulated_error",
            title="Recursive open-loop k-step accumulated error",
            ylabel="accumulated error",
        ),
        write_k_step_plot(
            artifact_path / "recursive_k_step_trajectory_rmse.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="recursive",
            value_name="trajectory_rmse",
            title="Recursive open-loop k-step trajectory RMSE",
            ylabel="trajectory RMSE",
        ),
        write_k_step_plot(
            artifact_path / "recursive_k_step_integral_square_error.png",
            method_summaries,
            methods=k_step_methods,
            metric_prefix="recursive",
            value_name="integral_square_error",
            title="Recursive open-loop k-step integral square error",
            ylabel="integral square error",
        ),
        write_grouped_method_bar_plot(
            artifact_path / "integrated_position_error.png",
            method_summaries,
            methods=representative_methods,
            title="Integrated trajectory position error",
            ylabel="integrated position mean error",
            metric="integrated_position_mean_error_mean",
            log_scale=True,
        ),
    ]
    summary_rows = flatten_plot_summary(method_summaries)
    write_rows_csv(artifact_path / "plot_summary.csv", summary_rows)
    payload = {
        "run_dirs": {scene: str(path) for scene, path in run_dirs.items()},
        "plots": [path.name for path in written],
        "include_zero_delta": include_zero_delta,
        "representative_methods": list(representative_methods),
        "fe_variant_methods": list(FE_VARIANT_METHODS),
        "k_step_methods": list(k_step_methods),
    }
    (artifact_path / "plot_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV artifact as dictionaries."""

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_all_methods_ranked_plot(
    path: Path,
    method_summaries: Mapping[str, list[dict[str, str]]],
) -> Path:
    """Write one horizontal ranked bar chart per scene."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    scenes = list(method_summaries)
    fig, axes = plt.subplots(
        1,
        len(scenes),
        figsize=(max(8, 5.2 * len(scenes)), 6),
        squeeze=False,
    )
    for ax, scene in zip(axes.ravel(), scenes):
        rows = sorted(
            method_summaries[scene],
            key=lambda row: _float(row, "mean_error_mean"),
            reverse=True,
        )
        labels = [_label(row) for row in rows]
        values = [_float(row, "mean_error_mean") for row in rows]
        colors = [METHOD_COLORS.get(row["method"], "#64748b") for row in rows]
        ax.barh(labels, values, color=colors)
        ax.set_xscale("log")
        ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(left=max(1e-3, min(values) * 0.75), right=max(values) * 1.25)
        ax.set_xlabel("mean one-step error")
        ax.set_title(scene)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("All baseline methods ranked by one-step error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_grouped_method_bar_plot(
    path: Path,
    method_summaries: Mapping[str, list[dict[str, str]]],
    *,
    methods: Iterable[str],
    title: str,
    ylabel: str,
    metric: str = "mean_error_mean",
    log_scale: bool,
) -> Path:
    """Write grouped bars with methods on x and scenes as adjacent bars."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(method_summaries)
    methods = [
        method
        for method in methods
        if any(_has_metric(_find(rows, method), metric) for rows in method_summaries.values())
    ]
    if not methods:
        path.write_text(f"{metric} unavailable\n")
        return path
    width = 0.78 / max(1, len(scenes))
    x_positions = list(range(len(methods)))
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(methods)), 4.8))
    for scene_index, scene in enumerate(scenes):
        rows = method_summaries[scene]
        offset = (scene_index - (len(scenes) - 1) / 2) * width
        values = [
            _float(_find(rows, method), metric)
            if _find(rows, method) is not None
            else 0.0
            for method in methods
        ]
        ax.bar(
            [pos + offset for pos in x_positions],
            values,
            width=width,
            label=scene,
            color=f"C{scene_index}",
            edgecolor="#111827",
            linewidth=0.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_method_label(method_summaries, method) for method in methods], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="holdout")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_per_window_plot(
    path: Path,
    window_metrics: Mapping[str, list[dict[str, str]]],
    *,
    methods: Iterable[str],
) -> Path:
    """Write per-window representative-method error trends."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(window_metrics)
    fig, axes = plt.subplots(
        1,
        len(scenes),
        figsize=(max(8, 5.4 * len(scenes)), 4.8),
        squeeze=False,
        sharey=True,
    )
    for ax, scene in zip(axes.ravel(), scenes):
        rows = window_metrics[scene]
        window_keys = _window_keys(rows)
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            if not method_rows:
                continue
            values = []
            for key in window_keys:
                row = next(
                    (
                        item
                        for item in method_rows
                        if _window_key(item) == key
                    ),
                    None,
                )
                values.append(float("nan") if row is None else _float(row, "mean_error"))
            ax.plot(
                range(len(window_keys)),
                values,
                marker="o",
                linewidth=1.6,
                linestyle=METHOD_STYLES.get(method, "-"),
                color=METHOD_COLORS.get(method, None),
                label=_method_label_from_rows(rows, method),
            )
        ax.set_yscale("log")
        ax.set_title(scene)
        ax.set_xlabel("mixed window index")
        ax.set_xticks(range(len(window_keys)))
        ax.set_xticklabels([str(index) for index in range(len(window_keys))])
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("mean one-step error")
    axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Representative method error across scene windows")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_k_step_plot(
    path: Path,
    method_summaries: Mapping[str, list[dict[str, str]]],
    *,
    methods: Iterable[str],
    metric_prefix: str,
    value_name: str,
    title: str,
    ylabel: str,
) -> Path:
    """Write horizon-error curves for logged or recursive k-step metrics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(method_summaries)
    fig, axes = plt.subplots(
        1,
        len(scenes),
        figsize=(max(8, 5.4 * len(scenes)), 4.8),
        squeeze=False,
        sharey=True,
    )
    for ax, scene in zip(axes.ravel(), scenes):
        rows = method_summaries[scene]
        for method in methods:
            row = _find(rows, method)
            if row is None:
                continue
            horizons, values = _k_step_values(row, metric_prefix, value_name)
            if not horizons:
                continue
            ax.plot(
                horizons,
                values,
                marker="o",
                linewidth=1.7,
                linestyle=METHOD_STYLES.get(method, "-"),
                color=METHOD_COLORS.get(method, None),
                label=_label(row),
            )
        ax.set_title(scene)
        ax.set_xlabel("horizon k")
        ax.set_xticks([1, 5, 10, 20, 50])
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel(ylabel)
    axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def flatten_plot_summary(
    method_summaries: Mapping[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Flatten key plot metrics into one small table."""

    rows: list[dict[str, object]] = []
    for scene, scene_rows in method_summaries.items():
        for row in scene_rows:
            rows.append(
                {
                    "scene": scene,
                    "method": row["method"],
                    "label": row["label"],
                    "mean_error_mean": _float(row, "mean_error_mean"),
                    "mean_rank": _float(row, "mean_rank"),
                    "win_count": _float(row, "win_count"),
                    "logged_k10_endpoint_error_mean": _float(
                        row,
                        "logged_k10_endpoint_error_mean",
                    ),
                    "logged_k10_accumulated_error_mean": _float(
                        row,
                        "logged_k10_accumulated_error_mean",
                    ),
                    "logged_k10_trajectory_rmse_mean": _float(
                        row,
                        "logged_k10_trajectory_rmse_mean",
                    ),
                    "logged_k10_integral_square_error_mean": _float(
                        row,
                        "logged_k10_integral_square_error_mean",
                    ),
                    "recursive_k10_final_step_error_mean": _float(
                        row,
                        "recursive_k10_final_step_error_mean",
                    ),
                    "recursive_k10_accumulated_error_mean": _float(
                        row,
                        "recursive_k10_accumulated_error_mean",
                    ),
                    "recursive_k10_trajectory_rmse_mean": _float(
                        row,
                        "recursive_k10_trajectory_rmse_mean",
                    ),
                    "recursive_k10_integral_square_error_mean": _float(
                        row,
                        "recursive_k10_integral_square_error_mean",
                    ),
                }
            )
    return rows


def write_rows_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    """Write rows as CSV."""

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find(rows: list[dict[str, str]], method: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("method") == method), None)


def _maybe_with_zero(methods: Iterable[str], include_zero_delta: bool) -> tuple[str, ...]:
    result = tuple(methods)
    if include_zero_delta and "zero_delta" not in result:
        return result + ("zero_delta",)
    return result


def _label(row: Mapping[str, str]) -> str:
    return str(row.get("label") or row.get("method"))


def _method_label(
    method_summaries: Mapping[str, list[dict[str, str]]],
    method: str,
) -> str:
    for rows in method_summaries.values():
        row = _find(rows, method)
        if row is not None:
            return _label(row)
    return method


def _method_label_from_rows(rows: list[dict[str, str]], method: str) -> str:
    row = _find(rows, method)
    return method if row is None else _label(row)


def _float(row: Mapping[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _has_metric(row: Mapping[str, str] | None, key: str) -> bool:
    return row is not None and str(row.get(key, "")) != ""


def _window_key(row: Mapping[str, str]) -> tuple[str, str, int, int]:
    return (
        str(row.get("split", "")),
        str(row.get("scene", "")),
        int(float(row.get("window_index", 0))),
        int(float(row.get("start_index", 0))),
    )


def _window_keys(rows: list[dict[str, str]]) -> list[tuple[str, str, int, int]]:
    keys: list[tuple[str, str, int, int]] = []
    for row in rows:
        key = _window_key(row)
        if key not in keys:
            keys.append(key)
    return keys


def _k_step_values(
    row: Mapping[str, str],
    metric_prefix: str,
    value_name: str,
) -> tuple[list[int], list[float]]:
    horizons = []
    values = []
    for horizon in (1, 5, 10, 20, 50):
        if metric_prefix == "logged":
            key = f"logged_k{horizon}_{value_name}_mean"
        elif metric_prefix == "recursive":
            key = f"recursive_k{horizon}_{value_name}_mean"
        else:
            raise ValueError(f"unknown metric prefix '{metric_prefix}'")
        if row.get(key, "") == "":
            continue
        horizons.append(horizon)
        values.append(float(row[key]))
    return horizons, values
