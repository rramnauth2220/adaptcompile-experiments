#!/usr/bin/env python3
"""Appendix Figure: Experiment 2 primary-predictor performance by objective.

This is an RQ2 diagnostic, not a program-selection-policy plot. It uses the
final FullAuto primary geometry predictor only and treats held-out episodes,
not episode-program rows, as the statistical unit for confidence intervals.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from plot_style import (
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    SEED,
    assert_episode_config_grid,
    bootstrap_ci,
    deterministic_jitter,
    ensure_dirs,
    json_script_metadata,
    panel_label,
    read_csv,
    read_json,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


PRIMARY_METHOD = "primary"
EXPECTED = {
    "geometry_mae": 0.0400217,
    "utility_correlation": 0.9620085,
    "mean_spearman": 0.80265,
    "pairwise_accuracy": 0.875,
}
OUTCOME_LABELS_FULL = {
    "acquisition": "Acquisition",
    "transfer": "Transfer",
    "boundedness": "Boundedness",
    "preservation": "Preservation",
}


def assert_close(name: str, observed: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{name} sanity check failed: observed={observed}, expected={expected}")


def primary_predicted_col(outcome: str) -> str:
    return f"{PRIMARY_METHOD}_predicted_{outcome}"


def validate_inputs(predictions: pd.DataFrame, ranking_by_episode: pd.DataFrame, ranking_json: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    if "meta_split" in predictions.columns:
        splits = set(predictions["meta_split"].astype(str))
        if splits != {"test"}:
            raise ValueError(f"Expected only TEST predictions, found splits {sorted(splits)}")
    if predictions["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 test episodes, found {predictions['episode_id'].nunique()}.")
    if len(predictions) != 400:
        raise ValueError(f"Expected 400 episode-program rows, found {len(predictions)}.")
    duplicate_count = int(predictions.duplicated(["episode_id", "config_id"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate episode/config rows in predictions: {duplicate_count}")
    assert_episode_config_grid(predictions, context="Experiment 2 primary predictions")

    counts = predictions.groupby("learning_type")["episode_id"].nunique().to_dict()
    bad_counts = {lt: counts.get(lt, 0) for lt in OBJECTIVE_ORDER if counts.get(lt, 0) != 20}
    if bad_counts:
        raise ValueError(f"Expected 20 test episodes per objective, got {bad_counts}")

    required = ["observed_utility", f"{PRIMARY_METHOD}_predicted_utility"]
    for outcome in OUTCOMES:
        required.extend([f"observed_{outcome}", primary_predicted_col(outcome)])
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction file missing required observed/predicted columns: {missing}")
    null_columns = [column for column in required if predictions[column].isna().any()]
    if null_columns:
        raise ValueError(f"Prediction file contains missing observed/predicted values: {null_columns}")

    observed_utility = predictions[[f"observed_{outcome}" for outcome in OUTCOMES]].astype(float).mean(axis=1)
    observed_diff = (predictions["observed_utility"].astype(float) - observed_utility).abs().max()
    if observed_diff > 1e-9:
        raise ValueError(f"Observed utility is not the equal-weight A/T/B/P mean; max diff={observed_diff}")
    predicted_utility = predictions[[primary_predicted_col(outcome) for outcome in OUTCOMES]].astype(float).mean(axis=1)
    predicted_diff = (predictions[f"{PRIMARY_METHOD}_predicted_utility"].astype(float) - predicted_utility).abs().max()
    if predicted_diff > 1e-9:
        raise ValueError(f"Primary predicted utility is not the equal-weight A/T/B/P mean; max diff={predicted_diff}")

    if metadata.get("objective_identity_used_in_primary_predictor") is not False:
        raise ValueError("Model metadata indicates objective identity was used by the primary predictor.")
    feature_names = [str(name).lower() for name in metadata.get("feature_names", [])]
    objective_features = [name for name in feature_names if "learning_type" in name or "objective" in name]
    if objective_features:
        raise ValueError(f"Objective-like feature names found in primary predictor: {objective_features[:5]}")

    required_ranking = {"episode_id", "learning_type", "method", "spearman", "pairwise_ranking_accuracy"}
    missing_ranking = required_ranking - set(ranking_by_episode.columns)
    if missing_ranking:
        raise ValueError(f"Ranking-by-episode file missing columns: {sorted(missing_ranking)}")
    primary = ranking_by_episode[ranking_by_episode["method"] == PRIMARY_METHOD]
    if primary["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 primary ranking episodes, found {primary['episode_id'].nunique()}.")
    if set(primary["episode_id"].astype(str)) != set(predictions["episode_id"].astype(str)):
        raise ValueError("Primary ranking audit episodes do not match prediction episodes.")
    if int(primary.duplicated(["episode_id", "method"]).sum()):
        raise ValueError("Duplicate primary ranking rows by episode.")
    if "primary" not in ranking_json:
        raise ValueError("Ranking JSON lacks primary predictor metrics.")


def build_episode_metrics(predictions: pd.DataFrame, ranking_by_episode: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    primary_ranking = ranking_by_episode[ranking_by_episode["method"] == PRIMARY_METHOD].set_index("episode_id")
    for episode_id, group in predictions.groupby("episode_id", sort=True):
        group = group.copy()
        learning_type = str(group["learning_type"].iloc[0])
        outcome_maes: Dict[str, float] = {}
        for outcome in OUTCOMES:
            outcome_maes[outcome] = float(
                (group[primary_predicted_col(outcome)].astype(float) - group[f"observed_{outcome}"].astype(float)).abs().mean()
            )
        row = {
            "episode_id": episode_id,
            "learning_type": learning_type,
            "objective": OBJECTIVE_LABELS[learning_type],
            "n_programs": int(group["config_id"].nunique()),
            "geometry_mae": float(np.mean([outcome_maes[outcome] for outcome in OUTCOMES])),
            "spearman": float(primary_ranking.loc[episode_id, "spearman"]),
            "pairwise_accuracy": float(primary_ranking.loc[episode_id, "pairwise_ranking_accuracy"]),
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_mae"] = outcome_maes[outcome]
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(episode_metrics: pd.DataFrame, predictions: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, lt in enumerate(OBJECTIVE_ORDER):
        sub = episode_metrics[episode_metrics["learning_type"] == lt].copy()
        if sub["episode_id"].nunique() != 20:
            raise ValueError(f"{lt} has {sub['episode_id'].nunique()} episode metrics, expected 20.")
        geom_mean, geom_lo, geom_hi = bootstrap_ci(sub["geometry_mae"].astype(float), seed=SEED + i)
        spear_mean, spear_lo, spear_hi = bootstrap_ci(sub["spearman"].astype(float), seed=SEED + 100 + i)
        pair_mean, pair_lo, pair_hi = bootstrap_ci(sub["pairwise_accuracy"].astype(float), seed=SEED + 200 + i)
        pred_sub = predictions[predictions["learning_type"] == lt]
        utility_corr = float(
            np.corrcoef(
                pred_sub["observed_utility"].astype(float),
                pred_sub[f"{PRIMARY_METHOD}_predicted_utility"].astype(float),
            )[0, 1]
        )
        row = {
            "learning_type": lt,
            "objective": OBJECTIVE_LABELS[lt],
            "n_episodes": int(sub["episode_id"].nunique()),
            "geometry_mae_mean": geom_mean,
            "geometry_mae_median": float(sub["geometry_mae"].astype(float).median()),
            "geometry_mae_ci_low": geom_lo,
            "geometry_mae_ci_high": geom_hi,
            "spearman_mean": spear_mean,
            "spearman_ci_low": spear_lo,
            "spearman_ci_high": spear_hi,
            "pairwise_accuracy_mean": pair_mean,
            "pairwise_accuracy_ci_low": pair_lo,
            "pairwise_accuracy_ci_high": pair_hi,
            "predicted_observed_utility_corr": utility_corr,
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_mae"] = float(sub[f"{outcome}_mae"].astype(float).mean())
        rows.append(row)
    return rows


def run_sanity_checks(
    predictions: pd.DataFrame,
    ranking_by_episode: pd.DataFrame,
    ranking_json: Mapping[str, Any],
    episode_metrics: pd.DataFrame,
) -> Dict[str, float]:
    aggregate_geometry_mae = float(episode_metrics["geometry_mae"].astype(float).mean())
    utility_corr = float(
        np.corrcoef(
            predictions["observed_utility"].astype(float),
            predictions[f"{PRIMARY_METHOD}_predicted_utility"].astype(float),
        )[0, 1]
    )
    primary_ranking = ranking_by_episode[ranking_by_episode["method"] == PRIMARY_METHOD]
    mean_spearman = float(primary_ranking["spearman"].astype(float).mean())
    pairwise_accuracy = float(primary_ranking["pairwise_ranking_accuracy"].astype(float).mean())

    assert_close("aggregate primary geometry MAE", aggregate_geometry_mae, EXPECTED["geometry_mae"])
    assert_close("primary utility correlation", utility_corr, EXPECTED["utility_correlation"])
    assert_close("primary mean Spearman", mean_spearman, EXPECTED["mean_spearman"])
    assert_close("primary pairwise accuracy", pairwise_accuracy, EXPECTED["pairwise_accuracy"])
    assert_close(
        "primary mean Spearman matches ranking JSON",
        mean_spearman,
        float(ranking_json[PRIMARY_METHOD]["mean_episode_spearman"]),
        tolerance=1e-10,
    )
    assert_close(
        "primary pairwise accuracy matches ranking JSON",
        pairwise_accuracy,
        float(ranking_json[PRIMARY_METHOD]["pairwise_ranking_accuracy"]),
        tolerance=1e-10,
    )
    return {
        "aggregate_geometry_mae": aggregate_geometry_mae,
        "primary_utility_correlation": utility_corr,
        "primary_mean_spearman": mean_spearman,
        "primary_pairwise_accuracy": pairwise_accuracy,
    }


def draw_figure(
    episode_metrics: pd.DataFrame,
    summary_rows: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, float],
    output_root: Path,
    inputs: Sequence[Path],
) -> Dict[str, Any]:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.95), gridspec_kw={"width_ratios": [1.05, 1.1]})

    x = np.arange(len(OBJECTIVE_ORDER), dtype=float)
    summary_by_lt = {str(row["learning_type"]): row for row in summary_rows}

    for i, lt in enumerate(OBJECTIVE_ORDER):
        sub = episode_metrics[episode_metrics["learning_type"] == lt]
        vals = sub["geometry_mae"].astype(float).to_numpy()
        jitter = deterministic_jitter(len(vals), width=0.12, seed=SEED + i)
        ax_a.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=15,
            color="#555555",
            alpha=0.42,
            linewidths=0,
            zorder=2,
        )
        row = summary_by_lt[lt]
        mean = float(row["geometry_mae_mean"])
        lo = float(row["geometry_mae_ci_low"])
        hi = float(row["geometry_mae_ci_high"])
        ax_a.errorbar(
            i,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            markersize=5.0,
            color="#000000",
            ecolor="#000000",
            elinewidth=1.1,
            capsize=3,
            zorder=5,
        )
    ax_a.axhline(0, color="#777777", linewidth=0.8)
    ax_a.set_xticks(x, [OBJECTIVE_LABELS[lt] for lt in OBJECTIVE_ORDER], rotation=24, ha="right")
    ax_a.set_ylabel("Geometry MAE")
    ax_a.set_title("Prediction error by objective", loc="left")
    ax_a.set_ylim(bottom=0)
    ax_a.grid(axis="y")
    ax_a.set_axisbelow(True)

    metric_specs = [
        ("spearman", "spearman_mean", "spearman_ci_low", "spearman_ci_high", "Spearman", "o", -0.12),
        (
            "pairwise_accuracy",
            "pairwise_accuracy_mean",
            "pairwise_accuracy_ci_low",
            "pairwise_accuracy_ci_high",
            "Pairwise",
            "D",
            0.12,
        ),
    ]
    metric_colors = {"Spearman": "#AA3377", "Pairwise": "#4477AA"}
    for metric_col, mean_col, lo_col, hi_col, label, marker, offset in metric_specs:
        for i, lt in enumerate(OBJECTIVE_ORDER):
            sub = episode_metrics[episode_metrics["learning_type"] == lt]
            vals = sub[metric_col].astype(float).to_numpy()
            jitter = deterministic_jitter(len(vals), width=0.045, seed=SEED + 300 + i + int(offset > 0))
            ax_b.scatter(
                np.full(len(vals), i + offset) + jitter,
                vals,
                s=11,
                color=metric_colors[label],
                alpha=0.22,
                linewidths=0,
                zorder=2,
            )
            row = summary_by_lt[lt]
            mean = float(row[mean_col])
            lo = float(row[lo_col])
            hi = float(row[hi_col])
            ax_b.errorbar(
                i + offset,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt=marker,
                markersize=5.0,
                color=metric_colors[label],
                ecolor=metric_colors[label],
                markeredgecolor="#333333",
                markeredgewidth=0.45,
                elinewidth=1.1,
                capsize=3,
                label=label if i == 0 else None,
                zorder=5,
            )
    ax_b.set_xticks(x, [OBJECTIVE_LABELS[lt] for lt in OBJECTIVE_ORDER], rotation=24, ha="right")
    ax_b.set_ylabel("Score")
    ax_b.set_title("Program-ranking fidelity by objective", loc="left")
    ax_b.set_ylim(-0.02, 1.04)
    ax_b.grid(axis="y")
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    fig.tight_layout(w_pad=2.0)

    dirs = ensure_dirs(output_root)
    episode_path = dirs["data"] / "exp2_by_objective_episode_metrics.csv"
    summary_path = dirs["data"] / "exp2_by_objective_summary.csv"
    episode_metrics.to_csv(episode_path, index=False)
    write_summary(summary_path, summary_rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "Experiment 2 appendix by-objective primary geometry prediction",
            "episode_metrics_csv": str(episode_path),
            "summary_csv": str(summary_path),
            "n_test_episodes": int(episode_metrics["episode_id"].nunique()),
            "n_bootstrap": N_BOOT,
            "statistical_unit": "episode",
            "primary_predictor_only": True,
            "objective_identity_used_in_primary_predictor": False,
            "sanity_checks": dict(sanity),
            "summary": list(summary_rows),
        }
    )
    save_outputs(fig, dirs["appendix"] / "fig_exp2_by_objective", metadata, dpi=300)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 2 by-objective appendix figure.")
    parser.add_argument("--predictions", default="artifacts/llama/prediction/predictions_test.csv")
    parser.add_argument("--ranking_json", default="artifacts/llama/prediction/ranking_metrics.json")
    parser.add_argument("--ranking_by_episode", default="artifacts/llama/prediction/ranking_metrics_by_episode.csv")
    parser.add_argument("--model_metadata", default="artifacts/llama/prediction/model_metadata.json")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    ranking_json_path = Path(args.ranking_json)
    ranking_by_episode_path = Path(args.ranking_by_episode)
    metadata_path = Path(args.model_metadata)
    predictions = read_csv(predictions_path)
    ranking_json = read_json(ranking_json_path)
    ranking_by_episode = read_csv(ranking_by_episode_path)
    model_metadata = read_json(metadata_path)

    validate_inputs(predictions, ranking_by_episode, ranking_json, model_metadata)
    episode_metrics = build_episode_metrics(predictions, ranking_by_episode)
    summary_rows = build_summary(episode_metrics, predictions)
    sanity = run_sanity_checks(predictions, ranking_by_episode, ranking_json, episode_metrics)
    draw_figure(
        episode_metrics,
        summary_rows,
        sanity,
        Path(args.output_root),
        [predictions_path, ranking_json_path, ranking_by_episode_path, metadata_path],
    )
    hardest = max(summary_rows, key=lambda row: float(row["geometry_mae_mean"]))
    print(
        "Appendix Exp2 by objective:",
        f"episodes={episode_metrics['episode_id'].nunique()}",
        f"aggregate_geometry_mae={sanity['aggregate_geometry_mae']:.6f}",
        f"utility_r={sanity['primary_utility_correlation']:.6f}",
        f"hardest_by_mae={hardest['objective']}:{float(hardest['geometry_mae_mean']):.6f}",
    )


if __name__ == "__main__":
    main()
