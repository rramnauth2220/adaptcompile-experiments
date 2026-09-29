#!/usr/bin/env python3
"""Main Figure 5: represented-family versus LOFO generalization.

This plot compares ordinary held-out episodes from represented learning families
against leave-one-family-out (LOFO) evaluation where the target family is absent
from both meta-training and validation. It uses saved final result files only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    SELECTOR_COLORS,
    assert_episode_config_grid,
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
    "represented_macro_mae": 0.0400217,
    "lofo_macro_mae": 0.33918,
    "lofo_macro_compiler_regret": 0.08513,
    "lofo_macro_global_regret": 0.06202,
}


def assert_close(name: str, observed: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{name} sanity check failed: observed={observed}, expected={expected}")


def primary_predicted_col(outcome: str) -> str:
    return f"{PRIMARY_METHOD}_predicted_{outcome}"


def validate_represented_predictions(predictions: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    if "meta_split" in predictions.columns:
        splits = set(predictions["meta_split"].astype(str))
        if splits != {"test"}:
            raise ValueError(f"Expected represented-family TEST predictions only, found splits {sorted(splits)}")
    if predictions["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 represented-family test episodes, found {predictions['episode_id'].nunique()}.")
    if len(predictions) != 400:
        raise ValueError(f"Expected 400 represented-family episode-program rows, found {len(predictions)}.")
    duplicated = int(predictions.duplicated(["episode_id", "config_id"]).sum())
    if duplicated:
        raise ValueError(f"Duplicate represented-family episode/config rows: {duplicated}")
    assert_episode_config_grid(predictions, context="represented-family Experiment 2 predictions")

    counts = predictions.groupby("learning_type")["episode_id"].nunique().to_dict()
    bad_counts = {lt: counts.get(lt, 0) for lt in OBJECTIVE_ORDER if counts.get(lt, 0) != 20}
    if bad_counts:
        raise ValueError(f"Expected 20 represented-family episodes per objective, got {bad_counts}")

    required = []
    for outcome in OUTCOMES:
        required.extend([f"observed_{outcome}", primary_predicted_col(outcome)])
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Represented-family predictions missing required columns: {missing}")
    null_columns = [column for column in required if predictions[column].isna().any()]
    if null_columns:
        raise ValueError(f"Represented-family predictions contain missing values: {null_columns}")
    if metadata.get("objective_identity_used_in_primary_predictor") is not False:
        raise ValueError("Represented-family metadata indicates objective identity was used by the primary predictor.")


def represented_mae_by_objective(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_id, group in predictions.groupby("episode_id", sort=True):
        learning_type = str(group["learning_type"].iloc[0])
        outcome_maes = []
        for outcome in OUTCOMES:
            mae = (
                group[primary_predicted_col(outcome)].astype(float)
                - group[f"observed_{outcome}"].astype(float)
            ).abs().mean()
            outcome_maes.append(float(mae))
        rows.append(
            {
                "episode_id": episode_id,
                "learning_type": learning_type,
                "represented_episode_geometry_mae": float(np.mean(outcome_maes)),
            }
        )
    episode_df = pd.DataFrame(rows)
    out = (
        episode_df.groupby("learning_type", as_index=False)
        .agg(
            represented_geometry_mae=("represented_episode_geometry_mae", "mean"),
            represented_n_episodes=("episode_id", "nunique"),
        )
    )
    out["objective"] = out["learning_type"].map(OBJECTIVE_LABELS)
    return out


def validate_lofo_summary(lofo_summary: pd.DataFrame, lofo_root: Path) -> None:
    expected = set(OBJECTIVE_ORDER + ["macro_average"])
    observed = set(lofo_summary["held_out_learning_type"].astype(str))
    missing = expected - observed
    if missing:
        raise ValueError(f"LOFO summary missing expected folds: {sorted(missing)}")
    extra = observed - expected
    if extra:
        raise ValueError(f"LOFO summary has unexpected rows: {sorted(extra)}")
    for lt in OBJECTIVE_ORDER:
        row = lofo_summary[lofo_summary["held_out_learning_type"] == lt].iloc[0]
        if int(row["n_test_episodes"]) != 20:
            raise ValueError(f"LOFO fold {lt} has n_test_episodes={row['n_test_episodes']}, expected 20.")
        metadata_path = lofo_root / lt / "model_metadata.json"
        metadata = read_json(metadata_path)
        if metadata.get("held_out_learning_type") != lt:
            raise ValueError(f"{metadata_path} held_out_learning_type mismatch.")
        if metadata.get("feature_set") != "episode":
            raise ValueError(f"{metadata_path} does not use the final episode-only LOFO representation.")
        if metadata.get("objective_identity_used_in_primary_predictor") is not False:
            raise ValueError(f"{metadata_path} indicates objective identity was used.")
        train_types = set(metadata.get("training_learning_types", []))
        valid_types = set(metadata.get("validation_learning_types", []))
        if lt in train_types or lt in valid_types:
            raise ValueError(f"LOFO fold {lt} appears in training or validation metadata.")


def build_panel_rows(represented: pd.DataFrame, lofo_summary: pd.DataFrame) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    represented_lookup = represented.set_index("learning_type").to_dict(orient="index")
    lofo_lookup = lofo_summary.set_index("held_out_learning_type").to_dict(orient="index")
    panel_a: List[Dict[str, Any]] = []
    for lt in OBJECTIVE_ORDER:
        rep = represented_lookup[lt]
        lofo = lofo_lookup[lt]
        panel_a.append(
            {
                "learning_type": lt,
                "objective": OBJECTIVE_LABELS[lt],
                "represented_n_episodes": int(rep["represented_n_episodes"]),
                "lofo_n_episodes": int(lofo["n_test_episodes"]),
                "represented_geometry_mae": float(rep["represented_geometry_mae"]),
                "lofo_geometry_mae": float(lofo["test_mean_mae"]),
                "mae_increase": float(lofo["test_mean_mae"] - rep["represented_geometry_mae"]),
            }
        )

    macro = lofo_lookup["macro_average"]
    represented_macro = float(np.mean([row["represented_geometry_mae"] for row in panel_a]))
    lofo_macro = float(macro["test_mean_mae"])
    panel_b: List[Dict[str, Any]] = []
    for lt in OBJECTIVE_ORDER + ["macro_average"]:
        row = lofo_lookup[lt]
        label = OBJECTIVE_LABELS.get(lt, "Macro")
        compiler_regret = float(row["compiler_oracle_regret"])
        global_regret = float(row["global_fixed_oracle_regret"])
        panel_b.append(
            {
                "held_out_learning_type": lt,
                "objective": label,
                "n_test_episodes": int(row["n_test_episodes"]),
                "compiler_mean_utility": float(row["compiler_mean_utility"]),
                "global_fixed_mean_utility": float(row["global_fixed_mean_utility"]),
                "oracle_mean_utility": float(row["oracle_mean_utility"]),
                "compiler_oracle_regret": compiler_regret,
                "global_fixed_oracle_regret": global_regret,
                "delta_regret": float(global_regret - compiler_regret),
            }
        )
    sanity = {
        "represented_macro_geometry_mae": represented_macro,
        "lofo_macro_geometry_mae": lofo_macro,
        "lofo_macro_compiler_regret": float(macro["compiler_oracle_regret"]),
        "lofo_macro_global_regret": float(macro["global_fixed_oracle_regret"]),
    }
    return panel_a, panel_b, sanity


def run_sanity_checks(panel_a: Sequence[Mapping[str, Any]], panel_b: Sequence[Mapping[str, Any]], sanity: Mapping[str, float]) -> None:
    if len(panel_a) != 5:
        raise ValueError(f"Expected five objective rows in Panel A, found {len(panel_a)}.")
    if len(panel_b) != 6:
        raise ValueError(f"Expected five objectives plus macro in Panel B, found {len(panel_b)}.")
    assert_close("represented-family macro geometry MAE", sanity["represented_macro_geometry_mae"], EXPECTED["represented_macro_mae"])
    assert_close("LOFO macro geometry MAE", sanity["lofo_macro_geometry_mae"], EXPECTED["lofo_macro_mae"])
    assert_close("LOFO macro compiler regret", sanity["lofo_macro_compiler_regret"], EXPECTED["lofo_macro_compiler_regret"])
    assert_close("LOFO macro global regret", sanity["lofo_macro_global_regret"], EXPECTED["lofo_macro_global_regret"])


def draw_figure(
    panel_a: Sequence[Mapping[str, Any]],
    panel_b: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, float],
    output_root: Path,
    inputs: Sequence[Path],
) -> Dict[str, Any]:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.95), gridspec_kw={"width_ratios": [1.05, 1.08]})

    y = np.arange(len(panel_a), dtype=float)
    represented_color = SELECTOR_COLORS["objective_conditioned_mean"]
    lofo_color = SELECTOR_COLORS["primary"]
    legend_handles = []
    for i, row in enumerate(panel_a):
        rep = float(row["represented_geometry_mae"])
        lofo = float(row["lofo_geometry_mae"])
        ax_a.plot([rep, lofo], [i, i], color="#9A9A9A", linewidth=1.0, zorder=1)
        rep_handle = ax_a.scatter(rep, i, marker="o", s=34, color=represented_color, edgecolor="#333333", linewidths=0.45, label="Represented family" if i == 0 else None, zorder=3)
        lofo_handle = ax_a.scatter(lofo, i, marker="D", s=38, color=lofo_color, edgecolor="#333333", linewidths=0.45, label="Held-out family (LOFO)" if i == 0 else None, zorder=4)
        if i == 0:
            legend_handles = [rep_handle, lofo_handle]
    ax_a.set_yticks(y, [str(row["objective"]) for row in panel_a])
    ax_a.set_ylim(len(panel_a) - 0.5, -0.5)
    ax_a.set_xlabel("Geometry MAE")
    ax_a.set_title("Represented vs. unseen families", loc="left")
    ax_a.grid(axis="x")
    ax_a.set_axisbelow(True)
    panel_label(ax_a, "A")

    x_positions = np.array([0, 1, 2, 3, 4, 5.65], dtype=float)
    deltas = np.array([float(row["delta_regret"]) for row in panel_b], dtype=float)
    colors = ["#4477AA" if value > 1e-12 else "#AA3377" if value < -1e-12 else "#999999" for value in deltas]
    ax_b.bar(x_positions, deltas, width=0.62, color=colors, alpha=0.88)
    ax_b.axhline(0, color="#333333", linewidth=1.0)
    ax_b.set_xticks(x_positions, [str(row["objective"]) for row in panel_b], rotation=27, ha="right")
    ax_b.set_ylabel("Regret reduction vs. global fixed")
    ax_b.set_title("Zero-shot selection benefit", loc="left")
    ymin = min(-0.095, float(deltas.min()) - 0.015)
    ymax = max(0.035, float(deltas.max()) + 0.015)
    ax_b.set_ylim(ymin, ymax)
    ax_b.grid(axis="y")
    ax_b.set_axisbelow(True)
    ax_b.text(
        0.02,
        0.96,
        "+ helps\n- hurts",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    panel_label(ax_b, "B")

    fig.legend(
        legend_handles,
        ["Represented family", "Held-out family (LOFO)"],
        loc="upper center",
        bbox_to_anchor=(0.36, 1.03),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.1)

    dirs = ensure_dirs(output_root)
    panel_a_path = dirs["data"] / "exp5_generalization_panel_a.csv"
    panel_b_path = dirs["data"] / "exp5_generalization_panel_b.csv"
    write_summary(panel_a_path, panel_a)
    write_summary(panel_b_path, panel_b)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "Experiment 5 main generalization figure",
            "panel_a_csv": str(panel_a_path),
            "panel_b_csv": str(panel_b_path),
            "sanity_checks": dict(sanity),
            "source_guard": {
                "represented_family": "final ordinary held-out Experiment 2 full_auto predictions",
                "lofo": "final episode-only leave-one-family-out summary and fold metadata",
                "existing_lofo_detail_figure_preserved": "artifacts/figures/appendix/llama/fig_lofo_generalization.pdf",
            },
        }
    )
    save_outputs(fig, dirs["main"] / "fig_exp5_generalization", metadata, dpi=300)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Main Figure 5: represented-family versus LOFO generalization.")
    parser.add_argument("--represented_predictions", default="artifacts/llama/prediction/predictions_test.csv")
    parser.add_argument("--represented_metadata", default="artifacts/llama/prediction/model_metadata.json")
    parser.add_argument("--lofo_summary", default="artifacts/llama/lofo/lofo_summary.csv")
    parser.add_argument("--lofo_root", default="artifacts/llama/lofo")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    represented_path = Path(args.represented_predictions)
    represented_metadata_path = Path(args.represented_metadata)
    lofo_summary_path = Path(args.lofo_summary)
    lofo_root = Path(args.lofo_root)
    represented_predictions = read_csv(represented_path)
    represented_metadata = read_json(represented_metadata_path)
    lofo_summary = read_csv(lofo_summary_path)

    validate_represented_predictions(represented_predictions, represented_metadata)
    validate_lofo_summary(lofo_summary, lofo_root)
    represented = represented_mae_by_objective(represented_predictions)
    panel_a, panel_b, sanity = build_panel_rows(represented, lofo_summary)
    run_sanity_checks(panel_a, panel_b, sanity)
    draw_figure(
        panel_a,
        panel_b,
        sanity,
        Path(args.output_root),
        [represented_path, represented_metadata_path, lofo_summary_path, lofo_root / "*/model_metadata.json"],
    )
    print(
        "Main Fig. 5:",
        f"represented_mae={sanity['represented_macro_geometry_mae']:.6f}",
        f"lofo_mae={sanity['lofo_macro_geometry_mae']:.6f}",
        f"delta_regret_macro={next(row for row in panel_b if row['held_out_learning_type'] == 'macro_average')['delta_regret']:.6f}",
    )


if __name__ == "__main__":
    main()
