"""Generate robust real-data comparison plots from baseline sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RUNS = {
    "Direct ice test": "outputs/eval/20260710T204229Z_jackal_paper_direct_ice_test_sweep_8window",
    "Direct train terrains": "outputs/eval/20260710T230604Z_jackal_paper_direct_representative_train_sweep",
    "Ice held out": "outputs/eval/20260710T172027Z_small_ugv_ice_holdout_test_sweep_8window",
}

METHODS = [
    "offline_fe",
    "fe_rls",
    "fe_bayes",
    "fe_prior_rls",
    "fe_prior_bayes",
    "fe_window_ls",
    "fe_kalman",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "alpaca_cold_start_online",
    "alpaca_online",
    "alpaca_prior_static",
    "offline_alpaca",
    "offline_linear",
    "linear_rls",
    "static_node",
    "maml_static",
    "maml_online",
]

METHOD_LABELS = {
    "offline_fe": "FE static",
    "fe_rls": "FE-RLS",
    "fe_bayes": "FE-Bayes",
    "fe_prior_rls": "FE prior RLS",
    "fe_prior_bayes": "FE prior Bayes",
    "fe_window_ls": "FE window LS",
    "fe_kalman": "FE Kalman",
    "neuralfly_rls": "NeuralFly RLS",
    "neuralfly_prior_rls": "NeuralFly prior RLS",
    "alpaca_cold_start_online": "ALPaCA cold-start",
    "alpaca_online": "ALPaCA online",
    "alpaca_prior_static": "ALPaCA prior static",
    "offline_alpaca": "ALPaCA offline",
    "offline_linear": "Linear offline",
    "linear_rls": "Linear RLS",
    "static_node": "NODE static",
    "maml_static": "MAML static",
    "maml_online": "MAML online",
}

METHOD_COLORS = {
    "offline_fe": "#4b5563",
    "fe_rls": "#0f766e",
    "fe_bayes": "#10b981",
    "fe_prior_rls": "#115e59",
    "fe_prior_bayes": "#047857",
    "fe_window_ls": "#2563eb",
    "fe_kalman": "#0891b2",
    "neuralfly_rls": "#d97706",
    "neuralfly_prior_rls": "#b45309",
    "alpaca_cold_start_online": "#15803d",
    "alpaca_online": "#16a34a",
    "alpaca_prior_static": "#86efac",
    "offline_alpaca": "#22c55e",
    "offline_linear": "#9ca3af",
    "linear_rls": "#6b7280",
    "static_node": "#dc2626",
    "maml_static": "#be185d",
    "maml_online": "#db2777",
}


def main() -> int:
    args = build_parser().parse_args()
    run_dirs = {
        label: Path(path)
        for label, path in parse_run_args(args.run).items()
    }
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)

    method_summaries = {
        label: load_rows(path / "method_summary.csv")
        for label, path in run_dirs.items()
    }
    scene_summaries = {
        label: load_rows(path / "scene_method_summary.csv")
        for label, path in run_dirs.items()
    }
    window_metrics = {
        label: load_rows(path / "window_metrics.csv")
        for label, path in run_dirs.items()
    }

    write_mean_error_plot(output_dir / "real_data_mean_error.png", method_summaries)
    write_metric_bars(
        output_dir / "real_data_trajectory_metrics.png",
        method_summaries,
        metrics=[
            ("adaptation_samples_to_25pct_improvement_mean", "Samples to 25% adaptation"),
            ("integrated_position_mean_error_mean", "Integrated position error"),
            ("logged_k10_trajectory_rmse_mean", "Logged 10-step trajectory RMSE"),
        ],
    )
    write_k_step_plot(
        output_dir / "real_data_logged_k_step_rmse.png",
        method_summaries,
        metric="trajectory_rmse",
    )
    write_scene_summary_plot(
        output_dir / "direct_jackal_scene_mean_error.png",
        scene_summaries["Direct train terrains"],
        title="Direct Jackal replication: representative terrains",
    )
    write_scene_summary_plot(
        output_dir / "direct_jackal_ice_scene_mean_error.png",
        scene_summaries["Direct ice test"],
        title="Direct Jackal replication: ice/autonomy test",
    )
    write_window_rank_plot(
        output_dir / "direct_jackal_ice_window_winners.png",
        window_metrics["Direct ice test"],
        title="Direct Jackal ice test: per-window winners",
    )
    write_summary(output_dir / "summary.json", run_dirs, method_summaries)
    print(output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Comparison run directory. Defaults to the latest direct/holdout runs.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def parse_run_args(values: list[str]) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_RUNS)
    runs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be LABEL=DIR, got {value!r}")
        label, path = value.split("=", maxsplit=1)
        runs[label] = path
    return runs


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs/eval") / f"{stamp}_real_data_status_plots"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def write_mean_error_plot(path: Path, summaries: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, len(summaries), figsize=(5.2 * len(summaries), 7), sharex=False)
    if len(summaries) == 1:
        axes = [axes]
    for ax, (label, rows) in zip(axes, summaries.items()):
        lookup = by_method(rows)
        values = []
        labels = []
        colors = []
        for method in METHODS:
            value = finite_float(lookup.get(method, {}).get("mean_error_mean"))
            if value is None:
                continue
            values.append(value)
            labels.append(METHOD_LABELS.get(method, method))
            colors.append(METHOD_COLORS.get(method, "#6b7280"))
        order = sorted(range(len(values)), key=values.__getitem__)[:8]
        ax.barh(
            [labels[index] for index in order],
            [values[index] for index in order],
            color=[colors[index] for index in order],
        )
        ax.set_title(label)
        ax.set_xlabel("Mean one-step error")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_metric_bars(
    path: Path,
    summaries: dict[str, list[dict[str, str]]],
    *,
    metrics: list[tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(len(metrics), len(summaries), figsize=(5.2 * len(summaries), 4.2 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    if len(summaries) == 1:
        axes = [[axis] for axis in axes]
    for row_axes, (metric, metric_label) in zip(axes, metrics):
        for ax, (run_label, rows) in zip(row_axes, summaries.items()):
            lookup = by_method(rows)
            values = []
            labels = []
            colors = []
            for method in METHODS:
                value = finite_float(lookup.get(method, {}).get(metric))
                if value is None:
                    continue
                values.append(value)
                labels.append(METHOD_LABELS.get(method, method))
                colors.append(METHOD_COLORS.get(method, "#6b7280"))
            order = sorted(range(len(values)), key=values.__getitem__)[:10]
            ax.barh(
                [labels[index] for index in order],
                [values[index] for index in order],
                color=[colors[index] for index in order],
            )
            ax.set_title(f"{run_label}\n{metric_label}")
            ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_k_step_plot(
    path: Path,
    summaries: dict[str, list[dict[str, str]]],
    *,
    metric: str,
) -> None:
    horizons = [1, 5, 10, 20, 50]
    methods = [
        "offline_fe",
        "fe_rls",
        "fe_bayes",
        "fe_window_ls",
        "neuralfly_rls",
        "alpaca_cold_start_online",
        "alpaca_online",
        "offline_alpaca",
        "offline_linear",
        "static_node",
        "maml_online",
    ]
    fig, axes = plt.subplots(1, len(summaries), figsize=(5.2 * len(summaries), 4.2), sharey=False)
    if len(summaries) == 1:
        axes = [axes]
    for ax, (run_label, rows) in zip(axes, summaries.items()):
        lookup = by_method(rows)
        for method in methods:
            ys = [
                finite_float(lookup.get(method, {}).get(f"logged_k{horizon}_{metric}_mean"))
                for horizon in horizons
            ]
            if all(value is None for value in ys):
                continue
            ax.plot(
                horizons,
                [float("nan") if value is None else value for value in ys],
                marker="o",
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, None),
            )
        ax.set_title(run_label)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Logged trajectory RMSE")
        ax.grid(alpha=0.25)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_scene_summary_plot(path: Path, rows: list[dict[str, str]], *, title: str) -> None:
    scenes = sorted({row["scene"] for row in rows})
    methods = [
        "offline_fe",
        "fe_rls",
        "fe_bayes",
        "fe_window_ls",
        "fe_kalman",
        "neuralfly_rls",
        "alpaca_cold_start_online",
        "alpaca_online",
        "offline_alpaca",
        "offline_linear",
        "static_node",
    ]
    fig, axes = plt.subplots(len(scenes), 1, figsize=(9, max(3.0, 2.2 * len(scenes))), sharex=False)
    if len(scenes) == 1:
        axes = [axes]
    for ax, scene in zip(axes, scenes):
        scene_rows = [row for row in rows if row["scene"] == scene]
        lookup = by_method(scene_rows)
        values = []
        labels = []
        colors = []
        for method in methods:
            value = finite_float(lookup.get(method, {}).get("mean_error_mean"))
            if value is None:
                continue
            values.append(value)
            labels.append(METHOD_LABELS.get(method, method))
            colors.append(METHOD_COLORS.get(method, "#6b7280"))
        order = sorted(range(len(values)), key=values.__getitem__)
        ax.barh(
            [labels[index] for index in order],
            [values[index] for index in order],
            color=[colors[index] for index in order],
        )
        ax.set_title(scene)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_window_rank_plot(path: Path, rows: list[dict[str, str]], *, title: str) -> None:
    winners: dict[str, int] = {}
    for key in sorted({(row["scene"], row["start_index"]) for row in rows}):
        candidates = [row for row in rows if (row["scene"], row["start_index"]) == key]
        best_method = None
        best_value = None
        for row in candidates:
            value = finite_float(row.get("mean_error"))
            if value is None:
                continue
            if best_value is None or value < best_value:
                best_value = value
                best_method = row["method"]
        if best_method is not None:
            winners[best_method] = winners.get(best_method, 0) + 1
    ordered = sorted(winners, key=winners.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [METHOD_LABELS.get(method, method) for method in ordered],
        [winners[method] for method in ordered],
        color=[METHOD_COLORS.get(method, "#6b7280") for method in ordered],
    )
    ax.set_title(title)
    ax.set_ylabel("Windows won")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(
    path: Path,
    run_dirs: dict[str, Path],
    summaries: dict[str, list[dict[str, str]]],
) -> None:
    payload = {"run_dirs": {label: str(run_dir) for label, run_dir in run_dirs.items()}, "top_methods": {}}
    for label, rows in summaries.items():
        finite_rows = [
            (row["method"], finite_float(row.get("mean_error_mean")))
            for row in rows
        ]
        finite_rows = [(method, value) for method, value in finite_rows if value is not None]
        finite_rows.sort(key=lambda item: item[1])
        payload["top_methods"][label] = [
            {"method": method, "mean_error_mean": value}
            for method, value in finite_rows[:8]
        ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
