#!/usr/bin/env python3
"""Main Figure 2: Experiment 2 geometry prediction.

The figure is regenerated entirely from saved Experiment 2 prediction artifacts.
Panel A and Panel B recompute utility correlation, utility MAE, and outcome MAEs
from predictions_test.csv. Panel C uses the saved ranking audit and verifies it
against the per-episode ranking table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from plot_style import (
    OUTCOMES,
    SELECTOR_COLORS,
    assert_episode_config_grid,
    ensure_dirs,
    format_number,
    json_script_metadata,
    panel_label,
    parse_bool,
    prettify_axes,
    read_csv,
    read_json,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


METHODS = ["configuration_mean", "objective_conditioned_mean", "primary"]
METHOD_LABELS = {
    "configuration_mean": "Configuration mean",
    "objective_conditioned_mean": "Objective-conditioned",
    "primary": "Primary predictor",
}
OUTCOME_LABELS_FULL = {
    "acquisition": "Acquisition",
    "transfer": "Transfer",
    "boundedness": "Boundedness",
    "preservation": "Preservation",
}
RANKING_METRICS = [
    ("mean_episode_spearman", "Rank corr."),
    ("pairwise_ranking_accuracy", "Pairwise"),
    ("top1_set_oracle_recovery", "Top-1"),
    ("top2_set_oracle_recovery", "Top-2"),
]

EXPECTED = {
    "primary_mean_mae": 0.0400217,
    "primary_mean_rmse": 0.0728361,
    "utility_correlation": {
        "primary": 0.9620085,
        "configuration_mean": 0.2973368,
        "objective_conditioned_mean": 0.9163082,
    },
    "configuration_mean_mae": 0.2381546,
    "objective_conditioned_mean_mae": 0.0597384,
    "primary_outcome_mae": {
        "acquisition": 0.0461558,
        "transfer": 0.0546833,
        "boundedness": 0.0565022,
        "preservation": 0.00274549,
    },
    "ranking": {
        "configuration_mean": {
            "mean_episode_spearman": 0.3605,
            "pairwise_ranking_accuracy": 0.6717,
            "top1_set_oracle_recovery": 0.41,
            "top2_set_oracle_recovery": 0.63,
        },
        "objective_conditioned_mean": {
            "mean_episode_spearman": 0.6955,
            "pairwise_ranking_accuracy": 0.8017,
            "top1_set_oracle_recovery": 0.64,
            "top2_set_oracle_recovery": 0.88,
        },
        "primary": {
            "mean_episode_spearman": 0.80265,
            "pairwise_ranking_accuracy": 0.875,
            "top1_set_oracle_recovery": 0.77,
            "top2_set_oracle_recovery": 0.93,
        },
    },
}


def assert_close(name: str, observed: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{name} sanity check failed: observed={observed}, expected={expected}")


def bool_mean(series: pd.Series) -> float:
    if series.dtype == bool:
        return float(series.mean())
    return float(series.map(parse_bool).mean())


def method_prediction_column(method: str, outcome: str) -> str:
    return f"{method}_predicted_{outcome}"


def validate_prediction_rows(predictions: pd.DataFrame, model_metadata: Mapping[str, Any]) -> None:
    if "meta_split" in predictions.columns:
        splits = set(predictions["meta_split"].astype(str))
        if splits != {"test"}:
            raise ValueError(f"Experiment 2 predictions must contain TEST rows only; found splits {sorted(splits)}")
    if predictions["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 test episodes, found {predictions['episode_id'].nunique()}.")
    if len(predictions) != 400:
        raise ValueError(f"Expected 400 episode-program prediction rows, found {len(predictions)}.")
    duplicated = int(predictions.duplicated(["episode_id", "config_id"]).sum())
    if duplicated:
        raise ValueError(f"Duplicate episode/config prediction rows: {duplicated}")
    assert_episode_config_grid(predictions, context="Experiment 2 predictions")

    required = ["observed_utility"]
    for outcome in OUTCOMES:
        required.append(f"observed_{outcome}")
        for method in METHODS:
            required.append(method_prediction_column(method, outcome))
        for method in METHODS:
            required.append(f"{method}_predicted_utility")
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")
    null_columns = [column for column in required if predictions[column].isna().any()]
    if null_columns:
        raise ValueError(f"Prediction file has missing observed/predicted values: {null_columns}")

    observed_expected = predictions[[f"observed_{outcome}" for outcome in OUTCOMES]].astype(float).mean(axis=1)
    observed_diff = (predictions["observed_utility"].astype(float) - observed_expected).abs().max()
    if observed_diff > 1e-9:
        raise ValueError(f"Observed balanced utility is not an equal-weight A/T/B/P mean; max diff={observed_diff}")
    for method in METHODS:
        expected_utility = predictions[[method_prediction_column(method, outcome) for outcome in OUTCOMES]].astype(float).mean(axis=1)
        utility_diff = (predictions[f"{method}_predicted_utility"].astype(float) - expected_utility).abs().max()
        if utility_diff > 1e-9:
            raise ValueError(f"{method} predicted utility is not an equal-weight A/T/B/P mean; max diff={utility_diff}")

    if model_metadata.get("objective_identity_used_in_primary_predictor") is not False:
        raise ValueError("Model metadata indicates objective identity was used in the primary predictor.")
    feature_names = [str(name).lower() for name in model_metadata.get("feature_names", [])]
    objective_like_features = [
        name for name in feature_names if "learning_type" in name or "objective" in name
    ]
    if objective_like_features:
        raise ValueError(f"Objective-like feature names found in primary predictor: {objective_like_features[:5]}")
    if model_metadata.get("objective_conditioned_mean_is_diagnostic_only") is not True:
        raise ValueError("Objective-conditioned baseline metadata is missing the diagnostic-only flag.")


def build_panel_a(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = predictions[
        [
            "episode_id",
            "learning_type",
            "config_id",
            "observed_utility",
            "primary_predicted_utility",
        ]
    ].copy()
    rows = rows.rename(columns={"primary_predicted_utility": "predicted_utility"})
    rows["absolute_utility_error"] = (rows["predicted_utility"].astype(float) - rows["observed_utility"].astype(float)).abs()
    utility_correlation_by_method = {}
    utility_mae_by_method = {}
    for method in METHODS:
        predicted = predictions[f"{method}_predicted_utility"].astype(float)
        observed = predictions["observed_utility"].astype(float)
        utility_correlation_by_method[method] = float(np.corrcoef(observed, predicted)[0, 1])
        utility_mae_by_method[method] = float((predicted - observed).abs().mean())
    return rows, {
        "primary_utility_correlation": utility_correlation_by_method["primary"],
        "primary_utility_mae": utility_mae_by_method["primary"],
        "utility_correlation_by_method": utility_correlation_by_method,
        "utility_mae_by_method": utility_mae_by_method,
    }


def build_panel_b(predictions: pd.DataFrame, metrics_json: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aggregate: Dict[str, Any] = {}
    for method in METHODS:
        method_maes = []
        method_rmses = []
        for outcome in OUTCOMES:
            observed = predictions[f"observed_{outcome}"].astype(float)
            predicted = predictions[method_prediction_column(method, outcome)].astype(float)
            error = predicted - observed
            mae = float(error.abs().mean())
            rmse = float(np.sqrt((error**2).mean()))
            method_maes.append(mae)
            method_rmses.append(rmse)
            json_mae = float(metrics_json[method]["by_outcome"][outcome]["mae"])
            json_rmse = float(metrics_json[method]["by_outcome"][outcome]["rmse"])
            assert_close(f"{method}.{outcome}.mae matches metrics JSON", mae, json_mae, tolerance=1e-10)
            assert_close(f"{method}.{outcome}.rmse matches metrics JSON", rmse, json_rmse, tolerance=1e-10)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS_FULL[outcome],
                    "mae": mae,
                    "rmse": rmse,
                }
            )
        aggregate[method] = {
            "mean_mae": float(np.mean(method_maes)),
            "mean_rmse": float(np.mean(method_rmses)),
        }
        assert_close(
            f"{method}.mean_mae matches metrics JSON",
            aggregate[method]["mean_mae"],
            float(metrics_json[method]["mean_mae"]),
            tolerance=1e-10,
        )
        assert_close(
            f"{method}.mean_rmse matches metrics JSON",
            aggregate[method]["mean_rmse"],
            float(metrics_json[method]["mean_rmse"]),
            tolerance=1e-10,
        )
    return rows, aggregate


def build_panel_c(ranking_metrics: Mapping[str, Any], ranking_by_episode: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    required = {"episode_id", "method", "spearman", "pairwise_ranking_accuracy", "top1_set_oracle_recovery", "top2_set_oracle_recovery"}
    missing = required - set(ranking_by_episode.columns)
    if missing:
        raise ValueError(f"Ranking-by-episode file missing columns: {sorted(missing)}")
    if ranking_by_episode["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 ranking episodes, found {ranking_by_episode['episode_id'].nunique()}.")
    rows: List[Dict[str, Any]] = []
    aggregate: Dict[str, Any] = {}
    for method in METHODS:
        sub = ranking_by_episode[ranking_by_episode["method"] == method].copy()
        if sub["episode_id"].nunique() != 100:
            raise ValueError(f"Ranking method {method} lacks 100 episodes.")
        computed = {
            "mean_episode_spearman": float(sub["spearman"].astype(float).mean()),
            "pairwise_ranking_accuracy": float(sub["pairwise_ranking_accuracy"].astype(float).mean()),
            "top1_set_oracle_recovery": bool_mean(sub["top1_set_oracle_recovery"]),
            "top2_set_oracle_recovery": bool_mean(sub["top2_set_oracle_recovery"]),
        }
        aggregate[method] = computed
        for key, value in computed.items():
            assert_close(f"{method}.{key} matches ranking JSON", value, float(ranking_metrics[method][key]), tolerance=1e-10)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": key,
                    "metric_label": dict(RANKING_METRICS)[key],
                    "score": value,
                    "n_episodes": int(sub["episode_id"].nunique()),
                }
            )
    return rows, aggregate


def run_final_sanity_checks(
    panel_a_stats: Mapping[str, float],
    panel_b_aggregate: Mapping[str, Mapping[str, float]],
    panel_b_rows: Sequence[Mapping[str, Any]],
    panel_c_aggregate: Mapping[str, Mapping[str, float]],
) -> None:
    for method, expected in EXPECTED["utility_correlation"].items():
        assert_close(
            f"{method} utility correlation",
            panel_a_stats["utility_correlation_by_method"][method],
            expected,
        )
    assert_close("primary geometry mean MAE", panel_b_aggregate["primary"]["mean_mae"], EXPECTED["primary_mean_mae"])
    assert_close("primary geometry mean RMSE", panel_b_aggregate["primary"]["mean_rmse"], EXPECTED["primary_mean_rmse"])
    assert_close("configuration mean aggregate MAE", panel_b_aggregate["configuration_mean"]["mean_mae"], EXPECTED["configuration_mean_mae"])
    assert_close(
        "objective-conditioned aggregate MAE",
        panel_b_aggregate["objective_conditioned_mean"]["mean_mae"],
        EXPECTED["objective_conditioned_mean_mae"],
    )
    row_lookup = {(str(row["method"]), str(row["outcome"])): float(row["mae"]) for row in panel_b_rows}
    for outcome, expected in EXPECTED["primary_outcome_mae"].items():
        assert_close(f"primary {outcome} MAE", row_lookup[("primary", outcome)], expected)
    for method, expected_metrics in EXPECTED["ranking"].items():
        for metric, expected in expected_metrics.items():
            assert_close(f"{method} {metric}", panel_c_aggregate[method][metric], expected)


def build_summaries(
    predictions_path: Path,
    metrics_json_path: Path,
    ranking_metrics_path: Path,
    ranking_by_episode_path: Path,
    model_metadata_path: Path,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    predictions = read_csv(predictions_path)
    metrics_json = read_json(metrics_json_path)
    ranking_metrics = read_json(ranking_metrics_path)
    ranking_by_episode = read_csv(ranking_by_episode_path)
    model_metadata = read_json(model_metadata_path)

    validate_prediction_rows(predictions, model_metadata)
    panel_a_rows, panel_a_stats = build_panel_a(predictions)
    panel_b_rows, panel_b_aggregate = build_panel_b(predictions, metrics_json)
    panel_c_rows, panel_c_aggregate = build_panel_c(ranking_metrics, ranking_by_episode)
    run_final_sanity_checks(panel_a_stats, panel_b_aggregate, panel_b_rows, panel_c_aggregate)

    summary = {
        "model_type": model_metadata.get("model_type"),
        "n_test_episodes": int(predictions["episode_id"].nunique()),
        "n_test_config_rows": int(len(predictions)),
        "primary_utility_correlation": panel_a_stats["primary_utility_correlation"],
        "primary_utility_mae": panel_a_stats["primary_utility_mae"],
        "utility_correlation_by_method": panel_a_stats["utility_correlation_by_method"],
        "utility_mae_by_method": panel_a_stats["utility_mae_by_method"],
        "geometry_aggregate": panel_b_aggregate,
        "ranking_aggregate": panel_c_aggregate,
        "objective_identity_used_in_primary_predictor": model_metadata.get("objective_identity_used_in_primary_predictor"),
        "objective_conditioned_mean_is_diagnostic_only": model_metadata.get("objective_conditioned_mean_is_diagnostic_only"),
    }
    return panel_a_rows, panel_b_rows, panel_c_rows, summary


def draw_figure(
    panel_a_rows: pd.DataFrame,
    panel_b_rows: Sequence[Mapping[str, Any]],
    panel_c_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_root: Path,
    inputs: Sequence[Path],
) -> dict:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.85),
        gridspec_kw={"width_ratios": [0.9, 1.2, 1.2]},
    )

    ax_a.scatter(
        panel_a_rows["observed_utility"],
        panel_a_rows["predicted_utility"],
        s=12,
        color=SELECTOR_COLORS["primary"],
        alpha=0.42,
        linewidths=0,
    )
    lo = min(float(panel_a_rows["observed_utility"].min()), float(panel_a_rows["predicted_utility"].min()))
    hi = max(float(panel_a_rows["observed_utility"].max()), float(panel_a_rows["predicted_utility"].max()))
    pad = 0.035 * (hi - lo)
    lo -= pad
    hi += pad
    ax_a.plot([lo, hi], [lo, hi], color="#333333", linewidth=0.9, linestyle="--")
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Observed utility")
    ax_a.set_ylabel("Predicted utility")
    ax_a.set_title("Predicted vs. observed\nutility", loc="left")
    ax_a.text(
        0.05,
        0.95,
        f"r = {summary['primary_utility_correlation']:.3f}\nn = {summary['n_test_config_rows']}",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
    )
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    method_offsets = {
        "configuration_mean": -0.18,
        "objective_conditioned_mean": 0.0,
        "primary": 0.18,
    }
    method_markers = {
        "configuration_mean": "s",
        "objective_conditioned_mean": "D",
        "primary": "o",
    }

    outcome_positions = np.arange(len(OUTCOMES), dtype=float)
    panel_b_lookup = {(str(row["method"]), str(row["outcome"])): row for row in panel_b_rows}
    legend_handles = []
    for method in METHODS:
        vals = [float(panel_b_lookup[(method, outcome)]["mae"]) for outcome in OUTCOMES]
        handle = ax_b.scatter(
            outcome_positions + method_offsets[method],
            vals,
            s=30 if method == "primary" else 24,
            color=SELECTOR_COLORS[method],
            marker=method_markers[method],
            alpha=0.95 if method == "primary" else 0.85,
            edgecolor="#333333" if method == "primary" else "none",
            linewidths=0.45 if method == "primary" else 0.0,
            label=METHOD_LABELS[method],
            zorder=4,
        )
        legend_handles.append(handle)
    ax_b.axhline(0, color="#777777", linewidth=0.8)
    ax_b.set_xticks(outcome_positions, [OUTCOME_LABELS_FULL[outcome] for outcome in OUTCOMES], rotation=28, ha="right")
    ax_b.set_ylabel("Mean absolute error")
    ax_b.set_title("Geometry prediction\nerror", loc="left")
    ax_b.set_ylim(bottom=0)
    prettify_axes(ax_b)
    panel_label(ax_b, "B")

    ranking_positions = np.arange(len(RANKING_METRICS), dtype=float)
    panel_c_lookup = {(str(row["method"]), str(row["metric"])): row for row in panel_c_rows}
    for method in METHODS:
        vals = [float(panel_c_lookup[(method, metric)]["score"]) for metric, _label in RANKING_METRICS]
        ax_c.scatter(
            ranking_positions + method_offsets[method],
            vals,
            s=30 if method == "primary" else 24,
            color=SELECTOR_COLORS[method],
            marker=method_markers[method],
            alpha=0.95 if method == "primary" else 0.85,
            edgecolor="#333333" if method == "primary" else "none",
            linewidths=0.45 if method == "primary" else 0.0,
            zorder=4,
        )
    ax_c.axhline(0, color="#777777", linewidth=0.8)
    ax_c.set_xticks(ranking_positions, [label for _metric, label in RANKING_METRICS], rotation=22, ha="right")
    ax_c.set_ylabel("Score")
    ax_c.set_title("Within-episode\nprogram ranking", loc="left")
    ax_c.set_ylim(0, 1.03)
    prettify_axes(ax_c)
    panel_label(ax_c, "C")

    fig.legend(
        legend_handles,
        [METHOD_LABELS[method] for method in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.9)

    dirs = ensure_dirs(output_root)
    panel_a_path = dirs["data"] / "exp2_prediction_panel_a.csv"
    panel_b_path = dirs["data"] / "exp2_prediction_panel_b.csv"
    panel_c_path = dirs["data"] / "exp2_prediction_panel_c.csv"
    panel_a_rows.to_csv(panel_a_path, index=False)
    write_summary(panel_b_path, panel_b_rows)
    write_summary(panel_c_path, panel_c_rows)

    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "Experiment 2 main prediction figure",
            "panel_a_csv": str(panel_a_path),
            "panel_b_csv": str(panel_b_path),
            "panel_c_csv": str(panel_c_path),
            **dict(summary),
        }
    )
    save_outputs(fig, dirs["main"] / "fig_exp2_prediction", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Main Figure 2: Experiment 2 geometry prediction.")
    parser.add_argument("--predictions", default="artifacts/llama/prediction/predictions_test.csv")
    parser.add_argument("--metrics_json", default="artifacts/llama/prediction/prediction_metrics.json")
    parser.add_argument("--ranking_json", default="artifacts/llama/prediction/ranking_metrics.json")
    parser.add_argument("--ranking_by_episode", default="artifacts/llama/prediction/ranking_metrics_by_episode.csv")
    parser.add_argument("--model_metadata", default="artifacts/llama/prediction/model_metadata.json")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    paths = [
        Path(args.predictions),
        Path(args.metrics_json),
        Path(args.ranking_json),
        Path(args.ranking_by_episode),
        Path(args.model_metadata),
    ]
    panel_a_rows, panel_b_rows, panel_c_rows, summary = build_summaries(*paths)
    metadata = draw_figure(panel_a_rows, panel_b_rows, panel_c_rows, summary, Path(args.output_root), paths)
    print(
        "Main Fig. 2:",
        f"n_points={metadata['n_test_config_rows']}",
        f"r={format_number(metadata['primary_utility_correlation'])}",
        f"utility_MAE={format_number(metadata['primary_utility_mae'])}",
        f"geometry_MAE={format_number(metadata['geometry_aggregate']['primary']['mean_mae'])}",
    )


if __name__ == "__main__":
    main()
