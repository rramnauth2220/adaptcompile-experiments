#!/usr/bin/env python3
"""Appendix Figure A3: outcome-level configuration sensitivity.

This figure decomposes balanced utility into its four behavioral outcomes for
the TEST episodes in the seed-aggregated geometry dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from plot_style import (
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    PROGRAM_COLORS,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    SEED,
    assert_balanced_utility,
    assert_test_episode_grid,
    bootstrap_ci,
    ensure_dirs,
    json_script_metadata,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


OUTCOME_TITLES = {
    "acquisition": "Acquisition",
    "transfer": "Transfer",
    "boundedness": "Boundedness",
    "preservation": "Preservation",
}


def load_test_geometry(geometry_path: Path):
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry, configs=PROGRAM_ORDER)
    test = geometry[geometry["meta_split"] == "test"].copy()
    test = test[test["config_id"].isin(PROGRAM_ORDER)].copy()
    missing_outcomes = [col for col in OUTCOMES if col not in test.columns or test[col].isna().any()]
    if missing_outcomes:
        raise ValueError(f"Missing outcome values in test geometry: {missing_outcomes}")
    counts = test.groupby("learning_type")["episode_id"].nunique().to_dict()
    bad_counts = {lt: counts.get(lt, 0) for lt in OBJECTIVE_ORDER if counts.get(lt, 0) != 20}
    if bad_counts:
        raise ValueError(f"Expected 20 TEST episodes per objective, got {bad_counts}")
    duplicate_count = int(test.duplicated(["episode_id", "config_id"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate TEST episode/config rows: {duplicate_count}")
    return test


def build_summary(test) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for outcome in OUTCOMES:
        for lt in OBJECTIVE_ORDER:
            for cfg in PROGRAM_ORDER:
                sub = test[(test["learning_type"] == lt) & (test["config_id"] == cfg)]
                vals = sub[outcome].astype(float).to_numpy()
                mean, lo, hi = bootstrap_ci(vals, seed=SEED + 100 * OUTCOMES.index(outcome) + PROGRAM_ORDER.index(cfg))
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": OUTCOME_TITLES[outcome],
                        "learning_type": lt,
                        "objective": OBJECTIVE_LABELS[lt],
                        "config_id": cfg,
                        "program": PROGRAM_LABELS[cfg],
                        "n_episodes": int(sub["episode_id"].nunique()),
                        "mean": mean,
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return rows


def draw_figure(test, summary_rows, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.95), sharex=True, sharey=True)
    axes = axes.ravel()
    y = np.arange(len(OBJECTIVE_ORDER), dtype=float)

    row_lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {
        (row["outcome"], row["learning_type"], row["config_id"]): row for row in summary_rows
    }
    offsets = np.linspace(-0.18, 0.18, len(PROGRAM_ORDER))
    for ax, outcome in zip(axes, OUTCOMES):
        for offset, cfg in zip(offsets, PROGRAM_ORDER):
            means = []
            lo_err = []
            hi_err = []
            for lt in OBJECTIVE_ORDER:
                row = row_lookup[(outcome, lt, cfg)]
                mean = float(row["mean"])
                means.append(mean)
                lo_err.append(mean - float(row["ci_low"]))
                hi_err.append(float(row["ci_high"]) - mean)
            ax.errorbar(
                means,
                y + offset,
                xerr=[lo_err, hi_err],
                color=PROGRAM_COLORS[cfg],
                marker="o",
                markersize=3.8,
                linestyle="none",
                linewidth=0,
                elinewidth=0.9,
                capsize=2.3,
                label=PROGRAM_LABELS[cfg],
                alpha=0.95,
            )
        ax.set_title(OUTCOME_TITLES[outcome], loc="left")
        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks(np.arange(0.0, 1.01, 0.25))
        ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.2f}")
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    for ax in axes:
        ax.set_yticks(y, [OBJECTIVE_LABELS[lt] for lt in OBJECTIVE_ORDER])
        ax.set_ylim(len(OBJECTIVE_ORDER) - 0.5, -0.5)
    for ax in axes[1::2]:
        ax.tick_params(axis="y", labelleft=False, length=0)
    for ax in axes[2:]:
        ax.set_xlabel("Outcome score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.4, h_pad=1.0)

    dirs = ensure_dirs(output_root)
    summary_path = dirs["data"] / "exp1_outcome_sensitivity_summary.csv"
    write_summary(summary_path, summary_rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "A3 outcome-level configuration sensitivity",
            "summary_csv": str(summary_path),
            "n_test_episodes": int(test["episode_id"].nunique()),
            "n_test_episodes_per_objective": {
                lt: int(test[test["learning_type"] == lt]["episode_id"].nunique()) for lt in OBJECTIVE_ORDER
            },
            "n_configs": len(PROGRAM_ORDER),
            "outcomes": OUTCOMES,
            "summary": summary_rows,
            "bootstrap_resamples": N_BOOT,
        }
    )
    outputs = save_outputs(fig, dirs["appendix"] / "fig_exp1_outcome_sensitivity", metadata)
    plt.close(fig)
    return outputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 1 appendix outcome-level sensitivity.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    geometry_path = Path(args.geometry)
    test = load_test_geometry(geometry_path)
    summary_rows = build_summary(test)
    draw_figure(test, summary_rows, Path(args.output_root), [geometry_path])
    print(
        "Appendix Exp1 outcome sensitivity:",
        f"episodes={test['episode_id'].nunique()}",
        f"rows={len(summary_rows)}",
    )


if __name__ == "__main__":
    main()
