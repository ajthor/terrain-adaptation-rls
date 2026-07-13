"""Create reviewer-facing Van der Pol comparison plots from method_summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_RUN_DIR = (
    "outputs/eval/20260713T201750Z_vdp_fe_bayes_alpaca_cold_comparison"
)

SCENARIO_LABELS = {
    "interpolation_mu_1.25": "Interpolation (mu=1.25)",
    "extrapolation_mu_3": "Extrapolation (mu=3.0)",
}
SCENARIO_ORDER = (
    "interpolation_mu_1.25",
    "extrapolation_mu_3",
)

METHOD_ORDER = (
    "fe_ode_rls",
    "fe_ode_bayes",
    "fe_ode_kalman",
    "fe_ode_window_ls",
    "fe_ode_sgd",
    "fe_ode_prior_rls",
    "fe_ode_prior_bayes",
    "fe_ode_static",
    "fe_mlp_rls",
    "alpaca_cold_start_online",
    "alpaca_online",
    "alpaca_static",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "neuralfly_static",
    "maml_online",
    "maml_static",
    "node_static",
)

COLD_START_METHODS = (
    "fe_ode_rls",
    "fe_ode_bayes",
    "fe_ode_kalman",
    "fe_ode_window_ls",
    "fe_ode_sgd",
    "fe_mlp_rls",
    "alpaca_cold_start_online",
    "neuralfly_rls",
    "maml_online",
    "maml_static",
    "node_static",
)

PRIOR_METHODS = (
    "fe_ode_prior_rls",
    "fe_ode_prior_bayes",
    "fe_ode_static",
    "alpaca_online",
    "alpaca_static",
    "neuralfly_prior_rls",
    "neuralfly_static",
    "maml_online",
    "maml_static",
    "node_static",
)

ONLINE_METHODS = (
    "fe_ode_rls",
    "fe_ode_bayes",
    "fe_ode_kalman",
    "fe_ode_window_ls",
    "fe_ode_sgd",
    "fe_ode_prior_rls",
    "fe_ode_prior_bayes",
    "fe_mlp_rls",
    "alpaca_cold_start_online",
    "alpaca_online",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "maml_online",
)

K_STEP_METHODS = (
    "fe_ode_rls",
    "fe_ode_bayes",
    "fe_ode_prior_rls",
    "fe_ode_prior_bayes",
    "fe_mlp_rls",
    "alpaca_cold_start_online",
    "alpaca_online",
    "neuralfly_rls",
    "neuralfly_prior_rls",
    "maml_online",
    "node_static",
)

FOCUSED_K_STEP_METHODS = (
    "fe_ode_rls",
    "fe_ode_bayes",
    "fe_ode_prior_rls",
    "fe_ode_prior_bayes",
    "alpaca_cold_start_online",
    "alpaca_online",
    "maml_online",
    "node_static",
)

METRIC_GRID = (
    ("mean_error", "One-step"),
    ("final_accumulated_error", "Accumulated"),
    ("recursive_k10_accumulated_error_mean", "k=10 rollout"),
    ("adaptation_samples_to_25pct_improvement", "Adapt samples"),
)

METHOD_COLORS = {
    "fe_ode_rls": "#0f766e",
    "fe_ode_bayes": "#10b981",
    "fe_ode_kalman": "#0891b2",
    "fe_ode_window_ls": "#2563eb",
    "fe_ode_sgd": "#7c3aed",
    "fe_ode_prior_rls": "#115e59",
    "fe_ode_prior_bayes": "#047857",
    "fe_ode_static": "#14b8a6",
    "fe_mlp_rls": "#6366f1",
    "alpaca_cold_start_online": "#15803d",
    "alpaca_online": "#16a34a",
    "alpaca_static": "#86efac",
    "neuralfly_rls": "#d97706",
    "neuralfly_prior_rls": "#b45309",
    "neuralfly_static": "#f97316",
    "maml_online": "#db2777",
    "maml_static": "#be185d",
    "node_static": "#dc2626",
    "zero_delta": "#111827",
}

METHOD_STYLES = {
    "fe_ode_rls": "-",
    "fe_ode_bayes": "--",
    "fe_ode_prior_rls": "-.",
    "fe_ode_prior_bayes": ":",
    "fe_mlp_rls": "-",
    "alpaca_cold_start_online": "--",
    "alpaca_online": "-",
    "neuralfly_rls": "--",
    "neuralfly_prior_rls": "-.",
    "maml_online": ":",
    "node_static": "-.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="VDP eval directory containing method_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reviewer plots. Defaults to RUN_DIR/vdp_reviewer_plots.",
    )
    parser.add_argument(
        "--include-zero-state-change",
        action="store_true",
        help="Include the zero state-change sanity check in the grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "vdp_reviewer_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_method_summary(run_dir / "method_summary.csv")
    if not args.include_zero_state_change:
        rows = [row for row in rows if row["method"] != "zero_delta"]
    rows_by_scenario = group_by_scenario(rows)

    written = [
        write_metric_heatmap(output_dir / "vdp_metric_heatmap.png", rows_by_scenario),
        write_ranked_metric_panels(
            output_dir / "vdp_cold_start_online.png",
            rows_by_scenario,
            COLD_START_METHODS,
            title="Cold-start and no-privileged-prior baselines",
            metric="mean_error",
            ylabel="mean one-step error",
            log_scale=True,
        ),
        write_ranked_metric_panels(
            output_dir / "vdp_prior_initialized.png",
            rows_by_scenario,
            PRIOR_METHODS,
            title="Prior-initialized and static baselines",
            metric="mean_error",
            ylabel="mean one-step error",
            log_scale=True,
        ),
        write_ranked_metric_panels(
            output_dir / "vdp_accumulated_trajectory_error.png",
            rows_by_scenario,
            COLD_START_METHODS,
            title="Cold-start accumulated trajectory error",
            metric="final_accumulated_error",
            ylabel="accumulated trajectory error",
            log_scale=True,
        ),
        write_adaptation_plot(output_dir / "vdp_adaptation_samples.png", rows_by_scenario),
        write_k_step_plot(
            output_dir / "vdp_recursive_k_step_accumulated_error.png",
            rows_by_scenario,
            methods=K_STEP_METHODS,
            metric_name="accumulated_error",
            ylabel="recursive accumulated error",
            title="Recursive open-loop rollout error",
        ),
        write_k_step_plot(
            output_dir / "vdp_recursive_k_step_final_error.png",
            rows_by_scenario,
            methods=K_STEP_METHODS,
            metric_name="final_step_error",
            ylabel="recursive endpoint error",
            title="Recursive open-loop endpoint error",
        ),
        write_k_step_plot(
            output_dir / "vdp_focused_k_step_accumulated_error.png",
            rows_by_scenario,
            methods=FOCUSED_K_STEP_METHODS,
            metric_name="accumulated_error",
            ylabel="recursive accumulated error",
            title="Focused recursive rollout error",
        ),
        write_k_step_plot(
            output_dir / "vdp_focused_k_step_final_error.png",
            rows_by_scenario,
            methods=FOCUSED_K_STEP_METHODS,
            metric_name="final_step_error",
            ylabel="recursive endpoint error",
            title="Focused recursive endpoint error",
        ),
    ]
    summary_rows = compact_summary(rows_by_scenario)
    write_csv(output_dir / "vdp_reviewer_summary.csv", summary_rows)
    manifest = {
        "source_run_dir": str(run_dir),
        "plots": [path.name for path in written],
        "summary_csv": "vdp_reviewer_summary.csv",
        "include_zero_state_change": args.include_zero_state_change,
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output_dir)
    return 0


def load_method_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_by_scenario(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    for scenario, scenario_rows in grouped.items():
        grouped[scenario] = sorted(
            scenario_rows,
            key=lambda row: (
                METHOD_ORDER.index(row["method"])
                if row["method"] in METHOD_ORDER
                else len(METHOD_ORDER),
                row["method"],
            ),
        )
    return {
        scenario: grouped[scenario]
        for scenario in SCENARIO_ORDER
        if scenario in grouped
    }


def write_metric_heatmap(
    path: Path,
    rows_by_scenario: Mapping[str, list[dict[str, str]]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(rows_by_scenario)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(max(10.0, 6.4 * len(scenarios)), 8.6),
        squeeze=False,
    )
    for ax, scenario in zip(axes.ravel(), scenarios):
        rows = rows_by_scenario[scenario]
        matrix = []
        annotations = []
        for row in rows:
            scores = []
            labels = []
            for metric, _ in METRIC_GRID:
                value = float_or_nan(row.get(metric, ""))
                metric_values = [
                    float_or_nan(item.get(metric, ""))
                    for item in rows
                    if is_finite(float_or_nan(item.get(metric, "")))
                ]
                scores.append(goodness_score(value, metric_values))
                labels.append(format_value(value))
            matrix.append(scores)
            annotations.append(labels)

        ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xticks(range(len(METRIC_GRID)))
        ax.set_xticklabels([label for _, label in METRIC_GRID], rotation=25, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["label"] for row in rows])
        ax.tick_params(length=0)
        for y, row in enumerate(rows):
            for x, text in enumerate(annotations[y]):
                color = "white" if matrix[y][x] < 0.28 else "#111827"
                ax.text(x, y, text, ha="center", va="center", fontsize=7.2, color=color)
            ax.text(
                -0.68,
                y,
                " ",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "square,pad=0.18",
                    "facecolor": METHOD_COLORS.get(row["method"], "#64748b"),
                    "edgecolor": "none",
                },
            )
        ax.set_xlim(-0.9, len(METRIC_GRID) - 0.5)
        ax.set_xlabel("lower is better; green marks the best values within a column")
        ax.grid(which="major", color="white", linewidth=1.0)
    fig.suptitle("Van der Pol baseline summary grid", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_ranked_metric_panels(
    path: Path,
    rows_by_scenario: Mapping[str, list[dict[str, str]]],
    methods: Iterable[str],
    *,
    title: str,
    metric: str,
    ylabel: str,
    log_scale: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(rows_by_scenario)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(max(10.0, 6.0 * len(scenarios)), 5.2),
        squeeze=False,
        sharey=False,
    )
    for ax, scenario in zip(axes.ravel(), scenarios):
        rows = [row for row in rows_by_scenario[scenario] if row["method"] in methods]
        rows = sorted(rows, key=lambda row: float_or_inf(row.get(metric, "")), reverse=True)
        labels = [row["label"] for row in rows]
        values = [float_or_nan(row.get(metric, "")) for row in rows]
        colors = [METHOD_COLORS.get(row["method"], "#64748b") for row in rows]
        ax.barh(labels, values, color=colors, edgecolor="#111827", linewidth=0.45)
        best = min((value for value in values if is_finite(value) and value > 0.0), default=0.0)
        for index, value in enumerate(values):
            if not is_finite(value):
                continue
            ax.text(value * (1.08 if log_scale else 1.01), index, format_value(value), va="center", fontsize=8)
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlim(left=max(best * 0.65, 1e-6))
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_adaptation_plot(
    path: Path,
    rows_by_scenario: Mapping[str, list[dict[str, str]]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(rows_by_scenario)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(max(10.0, 6.0 * len(scenarios)), 5.0),
        squeeze=False,
        sharey=False,
    )
    for ax, scenario in zip(axes.ravel(), scenarios):
        rows = [
            row
            for row in rows_by_scenario[scenario]
            if row["method"] in ONLINE_METHODS
            and float_or_nan(row.get("adaptation_reached_25pct_improvement", "")) > 0.5
        ]
        rows = sorted(
            rows,
            key=lambda row: float_or_inf(row.get("adaptation_samples_to_25pct_improvement", "")),
            reverse=True,
        )
        labels = [row["label"] for row in rows]
        values = [
            float_or_nan(row.get("adaptation_samples_to_25pct_improvement", ""))
            for row in rows
        ]
        colors = [METHOD_COLORS.get(row["method"], "#64748b") for row in rows]
        ax.barh(labels, values, color=colors, edgecolor="#111827", linewidth=0.45)
        for index, value in enumerate(values):
            ax.text(value + max(values) * 0.015, index, f"{int(round(value))}", va="center", fontsize=8)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel("samples to 25% online error reduction")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Adaptation speed from initial online error", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_k_step_plot(
    path: Path,
    rows_by_scenario: Mapping[str, list[dict[str, str]]],
    *,
    methods: Iterable[str],
    metric_name: str,
    ylabel: str,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(rows_by_scenario)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(max(10.0, 6.0 * len(scenarios)), 5.0),
        squeeze=False,
        sharey=True,
    )
    for ax, scenario in zip(axes.ravel(), scenarios):
        for method in methods:
            row = next((item for item in rows_by_scenario[scenario] if item["method"] == method), None)
            if row is None:
                continue
            horizons, values = k_step_values(row, metric_name)
            if not values:
                continue
            ax.plot(
                horizons,
                values,
                marker="o",
                linewidth=1.8,
                linestyle=METHOD_STYLES.get(method, "-"),
                color=METHOD_COLORS.get(method, "#64748b"),
                label=row["label"],
            )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel("recursive horizon k")
        ax.set_xticks([1, 5, 10, 25])
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel(ylabel)
    axes.ravel()[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def compact_summary(
    rows_by_scenario: Mapping[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, scenario_rows in rows_by_scenario.items():
        ranked = sorted(scenario_rows, key=lambda row: float_or_inf(row.get("mean_error", "")))
        ranks = {row["method"]: index + 1 for index, row in enumerate(ranked)}
        for row in scenario_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "method": row["method"],
                    "label": row["label"],
                    "mean_error_rank": ranks[row["method"]],
                    "mean_error": float_or_nan(row.get("mean_error", "")),
                    "final_accumulated_error": float_or_nan(
                        row.get("final_accumulated_error", "")
                    ),
                    "recursive_k10_accumulated_error_mean": float_or_nan(
                        row.get("recursive_k10_accumulated_error_mean", "")
                    ),
                    "adaptation_samples_to_25pct_improvement": float_or_nan(
                        row.get("adaptation_samples_to_25pct_improvement", "")
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def k_step_values(row: Mapping[str, str], metric_name: str) -> tuple[list[int], list[float]]:
    horizons: list[int] = []
    values: list[float] = []
    for horizon in (1, 5, 10, 25, 50):
        key = f"recursive_k{horizon}_{metric_name}_mean"
        value = float_or_nan(row.get(key, ""))
        if is_finite(value) and value > 0.0:
            horizons.append(horizon)
            values.append(value)
    return horizons, values


def goodness_score(value: float, values: list[float]) -> float:
    if not is_finite(value) or not values:
        return 0.0
    positive_values = [item for item in values if item > 0.0 and is_finite(item)]
    if value > 0.0 and positive_values:
        transformed = math.log10(value)
        transformed_values = [math.log10(item) for item in positive_values]
    else:
        transformed = value
        transformed_values = values
    low = min(transformed_values)
    high = max(transformed_values)
    if math.isclose(low, high):
        return 1.0
    return max(0.0, min(1.0, (high - transformed) / (high - low)))


def format_value(value: float) -> str:
    if not is_finite(value):
        return ""
    if abs(value) >= 1e4:
        return f"{value:.1e}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def float_or_nan(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def float_or_inf(value: str | None) -> float:
    result = float_or_nan(value)
    return result if is_finite(result) else float("inf")


def is_finite(value: float) -> bool:
    return math.isfinite(value)


if __name__ == "__main__":
    raise SystemExit(main())
