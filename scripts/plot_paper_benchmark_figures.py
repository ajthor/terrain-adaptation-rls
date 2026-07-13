"""Generate paper-facing benchmark plots from current evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_OUTPUT_DIR = "outputs/paper_figures"
DEFAULT_VDP_RUN = "outputs/eval/20260713T201750Z_vdp_fe_bayes_alpaca_cold_comparison"
DEFAULT_WARTY_SCENE1_RUN = (
    "outputs/eval/20260713T201739Z_warty_scene1_full_fe_bayes_alpaca_cold_plots"
)
DEFAULT_WARTY_SCENE5_RUN = (
    "outputs/eval/20260713T201737Z_warty_scene5_full_fe_bayes_alpaca_cold_plots"
)
DEFAULT_JACKAL_TRAIN_RUN = (
    "outputs/eval/20260713T204848Z_jackal_train_terrains_fe_bayes_alpaca_cold_256_plots"
)
DEFAULT_JACKAL_ICE_RUN = (
    "outputs/eval/20260713T204828Z_jackal_direct_ice_fe_bayes_alpaca_cold_256_plots"
)

REQUESTED_HORIZONS = (5, 10, 25)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    marker: str
    style: str = "-"


METHODS = (
    MethodSpec("fe_static", "FE static", "#64748b", "o", "--"),
    MethodSpec("fe_rls", "FE-RLS", "#0f766e", "o"),
    MethodSpec("fe_bayes", "FE-Bayes", "#10b981", "s"),
    MethodSpec("fe_kalman", "FE-Kalman", "#0891b2", "D"),
    MethodSpec("fe_window_ls", "FE-window LS", "#2563eb", "^"),
    MethodSpec("fe_sgd", "FE-SGD", "#7c3aed", "v"),
    MethodSpec("neuralfly", "NeuralFly", "#d97706", "P"),
    MethodSpec("alpaca_cold", "ALPaCA cold", "#15803d", "X"),
    MethodSpec("maml_online", "MAML", "#db2777", "*"),
    MethodSpec("node_static", "NODE", "#dc2626", "h", "-."),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}

VDP_METHOD_MAP = {
    "fe_ode_static": "fe_static",
    "fe_ode_rls": "fe_rls",
    "fe_ode_bayes": "fe_bayes",
    "fe_ode_kalman": "fe_kalman",
    "fe_ode_window_ls": "fe_window_ls",
    "fe_ode_sgd": "fe_sgd",
    "neuralfly_rls": "neuralfly",
    "alpaca_cold_start_online": "alpaca_cold",
    "maml_online": "maml_online",
    "node_static": "node_static",
}
REAL_METHOD_MAP = {
    "fe_prior_static": "fe_static",
    "fe_rls": "fe_rls",
    "fe_bayes": "fe_bayes",
    "fe_kalman": "fe_kalman",
    "fe_window_ls": "fe_window_ls",
    "fe_sgd": "fe_sgd",
    "neuralfly_rls": "neuralfly",
    "alpaca_cold_start_online": "alpaca_cold",
    "maml_online": "maml_online",
    "static_node": "node_static",
}


@dataclass(frozen=True)
class SummarySource:
    experiment: str
    split: str
    condition: str
    path: Path
    kind: str


@dataclass(frozen=True)
class WindowSource:
    experiment: str
    split: str
    condition: str
    path: Path
    kind: str


@dataclass(frozen=True)
class MetricRecord:
    experiment: str
    split: str
    condition: str
    method: str
    metric: str
    value: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vdp-run", default=DEFAULT_VDP_RUN)
    parser.add_argument("--warty-scene1-run", default=DEFAULT_WARTY_SCENE1_RUN)
    parser.add_argument("--warty-scene5-run", default=DEFAULT_WARTY_SCENE5_RUN)
    parser.add_argument("--jackal-train-run", default=DEFAULT_JACKAL_TRAIN_RUN)
    parser.add_argument("--jackal-ice-run", default=DEFAULT_JACKAL_ICE_RUN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    summary_sources = [
        SummarySource(
            "VDP",
            "held-out",
            "mu=1.25",
            Path(args.vdp_run) / "method_summary.csv",
            "vdp",
        ),
        SummarySource(
            "VDP",
            "held-out",
            "mu=3.0",
            Path(args.vdp_run) / "method_summary.csv",
            "vdp",
        ),
        SummarySource(
            "Warty",
            "held-out",
            "scene 1",
            Path(args.warty_scene1_run) / "method_summary.csv",
            "real",
        ),
        SummarySource(
            "Warty",
            "held-out",
            "scene 5",
            Path(args.warty_scene5_run) / "method_summary.csv",
            "real",
        ),
        SummarySource(
            "Jackal",
            "train",
            "train terrains",
            Path(args.jackal_train_run) / "method_summary.csv",
            "real",
        ),
        SummarySource(
            "Jackal",
            "held-out",
            "ice",
            Path(args.jackal_ice_run) / "method_summary.csv",
            "real",
        ),
    ]
    window_sources = [
        WindowSource(
            "Warty",
            "held-out",
            "scene 1",
            Path(args.warty_scene1_run) / "window_metrics.csv",
            "real",
        ),
        WindowSource(
            "Warty",
            "held-out",
            "scene 5",
            Path(args.warty_scene5_run) / "window_metrics.csv",
            "real",
        ),
        WindowSource(
            "Jackal",
            "train",
            "train terrains",
            Path(args.jackal_train_run) / "window_metrics.csv",
            "real",
        ),
        WindowSource(
            "Jackal",
            "held-out",
            "ice",
            Path(args.jackal_ice_run) / "window_metrics.csv",
            "real",
        ),
    ]
    online_sources = {
        "Warty scene 1": Path(args.warty_scene1_run) / "online_error_over_time.csv",
        "Warty scene 5": Path(args.warty_scene5_run) / "online_error_over_time.csv",
    }

    records = collect_summary_records(summary_sources)
    window_records = collect_window_records(window_sources)
    write_summary_csv(output_dir / "paper_metric_values.csv", records)
    write_coverage_grid(output_dir / "coverage_grid.png", records)
    write_metric_lines(
        output_dir / "mean_one_step_train.png",
        records,
        metric="one_step",
        split="train",
        ylabel="mean one-step error",
    )
    write_metric_lines(
        output_dir / "mean_one_step_heldout.png",
        records,
        metric="one_step",
        split="held-out",
        ylabel="mean one-step error",
    )
    write_metric_lines(
        output_dir / "trajectory_error_heldout.png",
        records,
        metric="trajectory",
        split="held-out",
        ylabel="mean trajectory error",
    )
    write_k_step_summary(
        output_dir / "mean_k_step_train.png",
        records,
        split="train",
        metric_prefix="k_endpoint",
        ylabel="mean k-step endpoint error",
    )
    write_k_step_summary(
        output_dir / "mean_k_step_heldout.png",
        records,
        split="held-out",
        metric_prefix="k_endpoint",
        ylabel="mean k-step endpoint error",
    )
    write_k_step_summary(
        output_dir / "accumulated_k_step_heldout.png",
        records,
        split="held-out",
        metric_prefix="k_accumulated",
        ylabel="accumulated k-step error",
    )
    write_window_k_step_plot(
        output_dir / "accumulated_k_step_over_windows_heldout.png",
        window_records,
        split="held-out",
        horizon=10,
    )
    write_vdp_mu_plot(output_dir / "vdp_performance_over_mu.png", records)
    write_warty_switching_plot(
        output_dir / "warty_error_over_time.png",
        online_sources,
    )
    write_experiment_specific_figures(output_dir, records, window_records, online_sources)
    print(output_dir)
    return 0


def write_experiment_specific_figures(
    output_dir: Path,
    records: list[MetricRecord],
    window_records: list[MetricRecord],
    online_sources: Mapping[str, Path],
) -> None:
    for experiment in ("VDP", "Warty", "Jackal"):
        experiment_dir = output_dir / experiment.lower()
        experiment_dir.mkdir()
        for split in ("train", "held-out"):
            if has_records(records, experiment=experiment, split=split, metric="one_step"):
                write_metric_lines(
                    experiment_dir / f"mean_one_step_{split_slug(split)}.png",
                    records,
                    metric="one_step",
                    split=split,
                    ylabel="mean one-step error",
                    experiment=experiment,
                )
            if split == "held-out" and has_records(
                records,
                experiment=experiment,
                split=split,
                metric="trajectory",
            ):
                write_metric_lines(
                    experiment_dir / "trajectory_error_heldout.png",
                    records,
                    metric="trajectory",
                    split=split,
                    ylabel="mean trajectory error",
                    experiment=experiment,
                )
            if has_k_records(records, experiment=experiment, split=split, prefix="k_endpoint"):
                write_k_step_summary(
                    experiment_dir / f"mean_k_step_{split_slug(split)}.png",
                    records,
                    split=split,
                    metric_prefix="k_endpoint",
                    ylabel="mean k-step endpoint error",
                    experiment=experiment,
                )
            if split == "held-out" and has_k_records(
                records,
                experiment=experiment,
                split=split,
                prefix="k_accumulated",
            ):
                write_k_step_summary(
                    experiment_dir / "accumulated_k_step_heldout.png",
                    records,
                    split=split,
                    metric_prefix="k_accumulated",
                    ylabel="accumulated k-step error",
                    experiment=experiment,
                )
        if experiment == "VDP":
            write_vdp_mu_plot(experiment_dir / "performance_over_mu.png", records)
        if experiment == "Warty":
            write_warty_switching_plot(experiment_dir / "error_over_time.png", online_sources)
            if has_window_records(window_records, experiment=experiment, split="held-out", horizon=10):
                write_window_k_step_plot(
                    experiment_dir / "accumulated_k_step_over_windows_heldout.png",
                    window_records,
                    split="held-out",
                    horizon=10,
                    experiment=experiment,
                )
        if experiment == "Jackal":
            if has_window_records(window_records, experiment=experiment, split="held-out", horizon=10):
                write_window_k_step_plot(
                    experiment_dir / "accumulated_k_step_over_windows_heldout.png",
                    window_records,
                    split="held-out",
                    horizon=10,
                    experiment=experiment,
                )


def collect_summary_records(sources: Iterable[SummarySource]) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for source in sources:
        if not source.path.is_file():
            continue
        rows = load_rows(source.path)
        if source.kind == "vdp":
            rows = [
                row
                for row in rows
                if vdp_condition(row.get("scenario", "")) == source.condition
            ]
        for row in rows:
            method = canonical_method(row["method"], source.kind)
            if method is None:
                continue
            for metric, value in metric_values(row, source.kind):
                records.append(
                    MetricRecord(
                        source.experiment,
                        source.split,
                        source.condition,
                        method,
                        metric,
                        value,
                    )
                )
    return records


def collect_window_records(sources: Iterable[WindowSource]) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for source in sources:
        if not source.path.is_file():
            continue
        for row in load_rows(source.path):
            method = canonical_method(row["method"], source.kind)
            if method is None:
                continue
            window_index = row.get("window_index", "")
            condition = f"{source.condition} window {window_index}"
            for horizon in REQUESTED_HORIZONS:
                value = finite_float(row.get(f"logged_k{horizon}_accumulated_error_mean"))
                if value is None:
                    continue
                records.append(
                    MetricRecord(
                        source.experiment,
                        source.split,
                        condition,
                        method,
                        f"k{horizon}_accumulated_window",
                        value,
                    )
                )
    return records


def metric_values(row: Mapping[str, str], kind: str) -> Iterable[tuple[str, float]]:
    one_step_key = "mean_error" if kind == "vdp" else "mean_error_mean"
    trajectory_key = (
        "final_accumulated_error"
        if kind == "vdp"
        else "integrated_position_mean_error_mean"
    )
    one_step = finite_float(row.get(one_step_key))
    if one_step is not None:
        yield "one_step", one_step
    trajectory = finite_float(row.get(trajectory_key))
    if trajectory is not None:
        yield "trajectory", trajectory
    for horizon in REQUESTED_HORIZONS:
        if kind == "vdp":
            endpoint_key = f"recursive_k{horizon}_final_step_error_mean"
            accumulated_key = f"recursive_k{horizon}_accumulated_error_mean"
        else:
            endpoint_key = f"logged_k{horizon}_endpoint_error_mean"
            accumulated_key = f"logged_k{horizon}_accumulated_error_mean"
        endpoint = finite_float(row.get(endpoint_key))
        if endpoint is not None:
            yield f"k{horizon}_endpoint", endpoint
        accumulated = finite_float(row.get(accumulated_key))
        if accumulated is not None:
            yield f"k{horizon}_accumulated", accumulated


def write_metric_lines(
    path: Path,
    records: list[MetricRecord],
    *,
    metric: str,
    split: str,
    ylabel: str,
    experiment: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt

    points = [
        record
        for record in records
        if record.metric == metric
        and record.split == split
        and (experiment is None or record.experiment == experiment)
    ]
    conditions = ordered_conditions(points)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.65 * len(conditions) + 2.8), 2.6))
    for method in METHODS:
        values = [lookup_value(points, condition, method.key) for condition in conditions]
        if all(value is None for value in values):
            continue
        ax.plot(
            range(len(conditions)),
            [float("nan") if value is None else value for value in values],
            color=method.color,
            marker=method.marker,
            linestyle=method.style,
            linewidth=1.5,
            markersize=3.4,
            label=method.label,
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    add_method_legend(fig, ncols=5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_k_step_summary(
    path: Path,
    records: list[MetricRecord],
    *,
    split: str,
    metric_prefix: str,
    ylabel: str,
    experiment: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt

    split_records = [
        record
        for record in records
        if record.split == split and (experiment is None or record.experiment == experiment)
    ]
    experiments = ordered_experiments(split_records)
    fig, axes = plt.subplots(
        1,
        max(1, len(experiments)),
        figsize=(max(3.6, 3.05 * max(1, len(experiments))), 2.65),
        sharey=True,
        squeeze=False,
    )
    for ax, experiment in zip(axes.ravel(), experiments):
        experiment_records = [
            record
            for record in split_records
            if record.experiment == experiment
        ]
        for method in METHODS:
            values = []
            for horizon in REQUESTED_HORIZONS:
                metric = f"k{horizon}_{metric_prefix.removeprefix('k_')}"
                value = mean_value(experiment_records, metric, method.key)
                values.append(value)
            if all(value is None for value in values):
                continue
            ax.plot(
                REQUESTED_HORIZONS,
                [float("nan") if value is None else value for value in values],
                color=method.color,
                marker=method.marker,
                linestyle=method.style,
                linewidth=1.5,
                markersize=3.4,
                label=method.label,
            )
        ax.text(
            0.02,
            0.96,
            experiment,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        ax.set_xticks(REQUESTED_HORIZONS)
        ax.set_xlabel("k")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel(ylabel)
    add_method_legend(fig, ncols=5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_window_k_step_plot(
    path: Path,
    records: list[MetricRecord],
    *,
    split: str,
    horizon: int,
    experiment: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt

    metric = f"k{horizon}_accumulated_window"
    points = [
        record
        for record in records
        if record.metric == metric
        and record.split == split
        and (experiment is None or record.experiment == experiment)
    ]
    experiments = ordered_experiments(points)
    fig, axes = plt.subplots(
        1,
        max(1, len(experiments)),
        figsize=(max(3.6, 3.1 * max(1, len(experiments))), 2.65),
        sharey=True,
        squeeze=False,
    )
    for ax, experiment in zip(axes.ravel(), experiments):
        experiment_points = [record for record in points if record.experiment == experiment]
        windows = ordered_window_conditions(experiment_points)
        for method in METHODS:
            values = [lookup_value(experiment_points, condition, method.key) for condition in windows]
            if all(value is None for value in values):
                continue
            ax.plot(
                range(len(windows)),
                [float("nan") if value is None else value for value in values],
                color=method.color,
                marker=method.marker,
                linestyle=method.style,
                linewidth=1.35,
                markersize=3.2,
                label=method.label,
            )
        ax.text(0.02, 0.96, experiment, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlabel("window")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel(f"accumulated k={horizon} error")
    add_method_legend(fig, ncols=5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_vdp_mu_plot(path: Path, records: list[MetricRecord]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt

    points = [
        record
        for record in records
        if record.experiment == "VDP" and record.split == "held-out" and record.metric == "one_step"
    ]
    mu_conditions = sorted(
        {
            (condition_mu(record.condition), display_condition(record))
            for record in points
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    for method in METHODS:
        values = []
        for _, condition in mu_conditions:
            values.append(lookup_value(points, condition, method.key))
        if all(value is None for value in values):
            continue
        ax.plot(
            [mu for mu, _ in mu_conditions],
            [float("nan") if value is None else value for value in values],
            color=method.color,
            marker=method.marker,
            linestyle=method.style,
            linewidth=1.5,
            markersize=3.4,
            label=method.label,
        )
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("mean one-step error")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    add_method_legend(fig, ncols=5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_warty_switching_plot(path: Path, sources: Mapping[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sources), figsize=(max(3.8, 3.25 * len(sources)), 2.55), sharey=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, (label, source) in zip(axes, sources.items()):
        rows = load_rows(source) if source.is_file() else []
        for method in METHODS:
            series = [
                row
                for row in rows
                if canonical_method(row.get("method", ""), "real") == method.key
            ]
            if not series:
                continue
            series = sorted(series, key=lambda row: finite_float(row.get("time_mean")) or 0.0)
            times = [finite_float(row.get("time_mean")) for row in series]
            values = [finite_float(row.get("cumulative_mean_error")) for row in series]
            filtered = [(time, value) for time, value in zip(times, values) if time is not None and value is not None]
            if not filtered:
                continue
            ax.plot(
                [time for time, _ in filtered],
                [value for _, value in filtered],
                color=method.color,
                marker=None,
                linestyle=method.style,
                linewidth=1.45,
                label=method.label,
            )
        ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("accumulated one-step error")
    add_method_legend(fig, ncols=5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_coverage_grid(path: Path, records: list[MetricRecord]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    set_paper_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    checks = [
        ("VDP train 1-step", "VDP", "train", "one_step"),
        ("VDP held-out 1-step", "VDP", "held-out", "one_step"),
        ("VDP held-out traj.", "VDP", "held-out", "trajectory"),
        ("VDP held-out k=25", "VDP", "held-out", "k25_endpoint"),
        ("Warty train 1-step", "Warty", "train", "one_step"),
        ("Warty held-out 1-step", "Warty", "held-out", "one_step"),
        ("Warty held-out traj.", "Warty", "held-out", "trajectory"),
        ("Warty held-out k=25", "Warty", "held-out", "k25_endpoint"),
        ("Jackal train 1-step", "Jackal", "train", "one_step"),
        ("Jackal held-out 1-step", "Jackal", "held-out", "one_step"),
        ("Jackal held-out traj.", "Jackal", "held-out", "trajectory"),
        ("Jackal held-out k=25", "Jackal", "held-out", "k25_endpoint"),
    ]
    matrix = []
    for method in METHODS:
        row_values = []
        for _, experiment, split, metric in checks:
            row_values.append(
                1
                if any(
                    record.experiment == experiment
                    and record.split == split
                    and record.metric == metric
                    and record.method == method.key
                    for record in records
                )
                else 0
            )
        matrix.append(row_values)
    fig, ax = plt.subplots(figsize=(9.8, 2.65))
    ax.imshow(matrix, cmap=ListedColormap(["#fca5a5", "#22c55e"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([label for label, *_ in checks], rotation=35, ha="right")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([method.label for method in METHODS])
    ax.tick_params(length=0)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, "ok" if value else "-", ha="center", va="center", fontsize=6.0)
    fig.legend(
        handles=[
            Patch(facecolor="#22c55e", label="available"),
            Patch(facecolor="#fca5a5", label="missing"),
        ],
        loc="outside upper center",
        ncols=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(path: Path, records: list[MetricRecord]) -> None:
    rows = [
        {
            "experiment": record.experiment,
            "split": record.split,
            "condition": record.condition,
            "method": record.method,
            "label": METHOD_BY_KEY[record.method].label,
            "metric": record.metric,
            "value": record.value,
        }
        for record in records
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "split", "condition", "method", "label", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def add_method_legend(fig, *, ncols: int) -> None:
    import matplotlib.pyplot as plt

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=method.color,
            marker=method.marker,
            linestyle=method.style,
            linewidth=1.5,
            markersize=3.6,
            label=method.label,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=ncols,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.7,
    )


def set_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.35,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def ordered_conditions(records: list[MetricRecord]) -> list[str]:
    order = {
        ("VDP", "mu=1.25"): 0,
        ("VDP", "mu=3"): 1,
        ("VDP", "mu=3.0"): 1,
        ("Warty", "scene 1"): 2,
        ("Warty", "scene 5"): 3,
        ("Jackal", "train terrains"): 4,
        ("Jackal", "ice"): 5,
    }
    keys = sorted(
        {(record.experiment, record.condition) for record in records},
        key=lambda item: (order.get(item, 100), item[0], item[1]),
    )
    return [condition if experiment in condition else f"{experiment} {condition}" for experiment, condition in keys]


def ordered_experiments(records: list[MetricRecord]) -> list[str]:
    order = {"VDP": 0, "Warty": 1, "Jackal": 2}
    return sorted({record.experiment for record in records}, key=lambda name: (order.get(name, 100), name))


def ordered_window_conditions(records: list[MetricRecord]) -> list[str]:
    def key(condition: str) -> tuple[str, int]:
        if " window " not in condition:
            return condition, 0
        prefix, index = condition.rsplit(" window ", maxsplit=1)
        try:
            return prefix, int(index)
        except ValueError:
            return prefix, 0

    return sorted({record.condition for record in records}, key=key)


def lookup_value(records: list[MetricRecord], condition: str, method: str) -> float | None:
    matches = [
        record.value
        for record in records
        if record.method == method
        and (record.condition == condition or display_condition(record) == condition)
    ]
    if not matches:
        return None
    return sum(matches) / len(matches)


def mean_value(records: list[MetricRecord], metric: str, method: str) -> float | None:
    values = [record.value for record in records if record.method == method and record.metric == metric]
    if not values:
        return None
    return sum(values) / len(values)


def display_condition(record: MetricRecord) -> str:
    if record.condition.startswith(record.experiment):
        return record.condition
    return f"{record.experiment} {record.condition}"


def has_records(
    records: list[MetricRecord],
    *,
    experiment: str,
    split: str,
    metric: str,
) -> bool:
    return any(
        record.experiment == experiment
        and record.split == split
        and record.metric == metric
        for record in records
    )


def has_k_records(
    records: list[MetricRecord],
    *,
    experiment: str,
    split: str,
    prefix: str,
) -> bool:
    suffix = prefix.removeprefix("k_")
    return any(
        record.experiment == experiment
        and record.split == split
        and record.metric == f"k{horizon}_{suffix}"
        for record in records
        for horizon in REQUESTED_HORIZONS
    )


def has_window_records(
    records: list[MetricRecord],
    *,
    experiment: str,
    split: str,
    horizon: int,
) -> bool:
    return any(
        record.experiment == experiment
        and record.split == split
        and record.metric == f"k{horizon}_accumulated_window"
        for record in records
    )


def split_slug(split: str) -> str:
    if split == "held-out":
        return "heldout"
    return split.replace("-", "_").replace(" ", "_")


def canonical_method(method: str, kind: str) -> str | None:
    mapping = VDP_METHOD_MAP if kind == "vdp" else REAL_METHOD_MAP
    return mapping.get(method)


def vdp_condition(scenario: str) -> str:
    if "mu_1.25" in scenario:
        return "mu=1.25"
    if "mu_3" in scenario:
        return "mu=3.0"
    return scenario.replace("_", " ")


def condition_mu(condition: str) -> float:
    return float(condition.split("=", maxsplit=1)[1])


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
