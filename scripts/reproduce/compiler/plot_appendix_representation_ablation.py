#!/usr/bin/env python3
"""Appendix Figure A: representation ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    FEATURE_SET_LABELS,
    SELECTOR_COLORS,
    ensure_dirs,
    format_number,
    json_script_metadata,
    panel_label,
    prediction_selected_utilities,
    prettify_axes,
    read_csv,
    read_json,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


FEATURE_DIRS = ["ablation_episode", "full_auto", "ablation_module_probes", "ablation_frozen_behavior"]
FEATURE_PATHS = {
    "ablation_episode": Path("ablations/episode"),
    "full_auto": Path("prediction"),
    "ablation_module_probes": Path("ablations/module_probes"),
    "ablation_frozen_behavior": Path("ablations/frozen_behavior"),
}


def collect_rows(experiment2_root: Path):
    rows = []
    inputs = []
    for feature_dir in FEATURE_DIRS:
        release_root = experiment2_root / FEATURE_PATHS[feature_dir]
        legacy_root = experiment2_root / feature_dir
        root = release_root if release_root.exists() else legacy_root
        metrics_path = root / "prediction_metrics.json"
        ranking_path = root / "ranking_metrics.json"
        predictions_path = root / "predictions_test.csv"
        if not (metrics_path.exists() and ranking_path.exists() and predictions_path.exists()):
            raise FileNotFoundError(f"Missing ablation files under {root}")
        inputs.extend([metrics_path, ranking_path, predictions_path])
        metrics = read_json(metrics_path)
        ranking = read_json(ranking_path)
        selected = prediction_selected_utilities(read_csv(predictions_path))
        rows.append(
            {
                "feature_dir": feature_dir,
                "feature_set": FEATURE_SET_LABELS[feature_dir],
                "mean_mae": metrics["primary"]["mean_mae"],
                "utility_correlation": metrics["utility_correlation"]["primary"],
                "mean_episode_spearman": ranking["primary"]["mean_episode_spearman"],
                "pairwise_ranking_accuracy": ranking["primary"]["pairwise_ranking_accuracy"],
                "top1_oracle_recovery": ranking["primary"]["top1_set_oracle_recovery"],
                "top2_oracle_recovery": ranking["primary"]["top2_set_oracle_recovery"],
                "selected_mean_utility": float(selected["selected_observed_utility"].mean()),
                "oracle_mean_utility": float(selected["oracle_observed_utility"].mean()),
                "compiler_oracle_regret": float(selected["oracle_regret"].mean()),
            }
        )
    return rows, inputs


def draw_figure(rows, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.65))
    labels = [row["feature_set"] for row in rows]
    x = np.arange(len(rows))

    ax_a.bar(x, [row["mean_mae"] for row in rows], color=SELECTOR_COLORS["primary"], alpha=0.82)
    ax_a.set_xticks(x, labels)
    ax_a.set_ylabel("Mean MAE")
    ax_a.set_title("Geometry prediction", loc="left")
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    ax_b.plot(
        x,
        [row["compiler_oracle_regret"] for row in rows],
        marker="o",
        color=SELECTOR_COLORS["primary"],
        linewidth=1.3,
    )
    for i, row in enumerate(rows):
        ax_b.text(i, row["compiler_oracle_regret"] + 0.002, f"{100*row['top1_oracle_recovery']:.0f}%", ha="center", fontsize=7.5)
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Compiler oracle regret")
    ax_b.set_title("Selection quality", loc="left")
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(w_pad=2.0)

    dirs = ensure_dirs(output_root)
    write_summary(dirs["data"] / "representation_ablation_summary.csv", rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"summary": rows, "top1_annotations": "Percent labels show primary top-1 oracle recovery."})
    save_outputs(fig, dirs["appendix"] / "fig_ablation_representation", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appendix representation ablation figure.")
    parser.add_argument("--experiment2_root", default="artifacts/llama")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    rows, inputs = collect_rows(Path(args.experiment2_root))
    metadata = draw_figure(rows, Path(args.output_root), inputs)
    best = min(metadata["summary"], key=lambda r: r["mean_mae"])
    print("Appendix representation:", f"best_mae={best['feature_set']} {format_number(best['mean_mae'])}")


if __name__ == "__main__":
    main()
