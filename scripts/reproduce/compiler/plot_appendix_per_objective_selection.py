#!/usr/bin/env python3
"""Appendix Figure B: per-objective selection behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    SELECTOR_COLORS,
    SELECTOR_LABELS,
    SELECTOR_MARKERS,
    assert_balanced_utility,
    assert_test_episode_grid,
    ensure_dirs,
    json_script_metadata,
    panel_label,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
    episode_selector_utilities,
)


def build_rows(geometry_path: Path, compiler_csv: Path):
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry)
    detail = episode_selector_utilities(geometry, compiler_csv=compiler_csv)
    rows = []
    for lt in OBJECTIVE_ORDER:
        sub = detail[detail["learning_type"] == lt]
        for selector in ["global_fixed", "objective_fixed", "compiler"]:
            oracle_hits = []
            for _, row in sub.iterrows():
                oracle = set(str(row["oracle_config_set"]).split("|"))
                selected = str(row[f"{selector}_config_id"])
                oracle_hits.append(selected in oracle)
            rows.append(
                {
                    "learning_type": lt,
                    "objective": OBJECTIVE_LABELS[lt],
                    "selector": selector,
                    "selector_label": SELECTOR_LABELS[selector],
                    "oracle_regret": float(sub[f"{selector}_regret"].mean()),
                    "top1_oracle_recovery": float(np.mean(oracle_hits)),
                    "mean_utility": float(sub[f"{selector}_utility"].mean()),
                    "n_episodes": int(sub["episode_id"].nunique()),
                }
            )
    return rows, detail


def draw_figure(rows, detail, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.75, 4.45), sharex=True)
    x = np.arange(len(OBJECTIVE_ORDER))
    offsets = {"global_fixed": -0.18, "objective_fixed": 0.0, "compiler": 0.18}
    selectors = ["global_fixed", "objective_fixed", "compiler"]

    for selector in selectors:
        vals = [
            next(row["oracle_regret"] for row in rows if row["learning_type"] == lt and row["selector"] == selector)
            for lt in OBJECTIVE_ORDER
        ]
        ax_a.scatter(
            x + offsets[selector],
            vals,
            marker=SELECTOR_MARKERS[selector],
            color=SELECTOR_COLORS[selector],
            s=34,
            label=SELECTOR_LABELS[selector],
            zorder=3,
        )
        ax_a.plot(x + offsets[selector], vals, color=SELECTOR_COLORS[selector], linewidth=0.8, alpha=0.65)
    ax_a.set_ylabel("Oracle regret")
    ax_a.set_title("Per-objective regret", loc="left")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    for selector in selectors:
        vals = [
            next(row["top1_oracle_recovery"] for row in rows if row["learning_type"] == lt and row["selector"] == selector)
            for lt in OBJECTIVE_ORDER
        ]
        ax_b.scatter(
            x + offsets[selector],
            vals,
            marker=SELECTOR_MARKERS[selector],
            color=SELECTOR_COLORS[selector],
            s=34,
            zorder=3,
        )
        ax_b.plot(x + offsets[selector], vals, color=SELECTOR_COLORS[selector], linewidth=0.8, alpha=0.65)
    ax_b.set_ylabel("Top-1 recovery")
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_xticks(x, [OBJECTIVE_LABELS[lt] for lt in OBJECTIVE_ORDER])
    ax_b.set_title("Oracle-optimal selection rate", loc="left")
    ax_b.yaxis.set_major_formatter(lambda value, _pos: f"{int(round(100*value))}%")
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(h_pad=2.1)

    dirs = ensure_dirs(output_root)
    write_summary(dirs["data"] / "per_objective_selection_summary.csv", rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"summary": rows, "n_test_episodes": int(detail["episode_id"].nunique())})
    save_outputs(fig, dirs["appendix"] / "fig_selection_by_objective", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appendix per-objective selection figure.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--compiler_csv", default="artifacts/llama/selection/compiler_evaluation.csv")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    rows, detail = build_rows(Path(args.geometry), Path(args.compiler_csv))
    draw_figure(rows, detail, Path(args.output_root), [Path(args.geometry), Path(args.compiler_csv)])
    print("Appendix per-objective selection:", f"n_test={detail['episode_id'].nunique()}")


if __name__ == "__main__":
    main()
