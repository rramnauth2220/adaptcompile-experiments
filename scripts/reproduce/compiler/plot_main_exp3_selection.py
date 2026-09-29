#!/usr/bin/env python3
"""Main Figure 3: Experiment 3 program selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    N_BOOT,
    SELECTOR_COLORS,
    SELECTOR_LABELS,
    SELECTOR_LINESTYLES,
    SELECTOR_MARKERS,
    SEED,
    UTILITY_SPECS,
    assert_balanced_utility,
    assert_test_episode_grid,
    bootstrap_ci,
    deterministic_jitter,
    ensure_dirs,
    episode_selector_utilities,
    format_number,
    json_script_metadata,
    panel_label,
    prettify_axes,
    read_csv,
    read_json,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


def utility_spec_path(root: Path, spec: str, suffix: str) -> Path:
    return root / "utility_specs" / f"{spec}.{suffix}"


def build_balanced_selection(geometry_path: Path, compiler_csv: Path):
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry)
    rows = episode_selector_utilities(geometry, compiler_csv=compiler_csv)
    if rows["episode_id"].nunique() != 100:
        raise ValueError("Main Experiment 3 requires 100 test episodes.")
    return geometry, rows


def build_utility_spec_summary(geometry, experiment3_root: Path):
    rows = []
    for spec, label in UTILITY_SPECS.items():
        csv_path = utility_spec_path(experiment3_root, spec, "csv")
        json_path = utility_spec_path(experiment3_root, spec, "json")
        summary = read_json(json_path)
        weights = summary["utility_weights"]
        utility_rows = episode_selector_utilities(geometry, weights=weights, compiler_csv=csv_path)
        rows.extend(
            [
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "selector": "global_fixed",
                    "selector_label": SELECTOR_LABELS["global_fixed"],
                    "oracle_regret": float(utility_rows["global_fixed_regret"].mean()),
                    "top1_recovery": np.nan,
                    "top2_recovery": np.nan,
                },
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "selector": "objective_fixed",
                    "selector_label": SELECTOR_LABELS["objective_fixed"],
                    "oracle_regret": float(utility_rows["objective_fixed_regret"].mean()),
                    "top1_recovery": np.nan,
                    "top2_recovery": np.nan,
                },
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "selector": "compiler",
                    "selector_label": SELECTOR_LABELS["compiler"],
                    "oracle_regret": float(summary["mean_oracle_regret"]),
                    "top1_recovery": float(summary["oracle_recovery_rate"]),
                    "top2_recovery": float(summary["top_k_recovery_rate"]),
                },
            ]
        )
    return rows


def draw_figure(selection_rows, utility_spec_rows, output_root: Path, inputs: list[Path]) -> dict:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.9), gridspec_kw={"width_ratios": [1.0, 1.1]})

    selectors = ["global_fixed", "objective_fixed", "compiler", "oracle"]
    y_columns = {
        "global_fixed": "global_fixed_utility",
        "objective_fixed": "objective_fixed_utility",
        "compiler": "compiler_utility",
        "oracle": "oracle_utility",
    }
    x = np.arange(len(selectors))
    summary_rows = []
    for i, selector in enumerate(selectors):
        vals = selection_rows[y_columns[selector]].to_numpy(float)
        jitter = deterministic_jitter(len(vals), width=0.08, seed=SEED + i)
        ax_a.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=10,
            color=SELECTOR_COLORS[selector],
            alpha=0.22 if selector != "oracle" else 0.12,
            linewidths=0,
        )
        mean, lo, hi = bootstrap_ci(vals)
        summary_rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "mean_utility": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "oracle_regret": float(selection_rows[f"{selector}_regret"].mean()) if selector != "oracle" else 0.0,
            }
        )
        ax_a.errorbar(
            i,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt=SELECTOR_MARKERS[selector],
            color=SELECTOR_COLORS[selector],
            markersize=5.5 if selector != "oracle" else 7.5,
            capsize=3,
            linewidth=1.1,
            zorder=5,
        )
    ax_a.plot(x, [row["mean_utility"] for row in summary_rows], color="#444444", linewidth=0.8, alpha=0.7)
    ax_a.set_xticks(x, ["Global\nfixed", "Objective\nfixed", "Compiler", "Oracle"])
    ax_a.set_ylabel("Realized balanced utility")
    ax_a.set_title("Selected program utility", loc="left")
    for i, row in enumerate(summary_rows):
        ax_a.text(i, ax_a.get_ylim()[0], f"regret\n{row['oracle_regret']:.3f}", ha="center", va="bottom", fontsize=7.2)
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    specs = list(UTILITY_SPECS)
    spec_x = np.arange(len(specs))
    for selector in ["global_fixed", "objective_fixed", "compiler"]:
        vals = [
            next(row["oracle_regret"] for row in utility_spec_rows if row["utility_spec"] == spec and row["selector"] == selector)
            for spec in specs
        ]
        ax_b.plot(
            spec_x,
            vals,
            marker=SELECTOR_MARKERS[selector],
            linestyle=SELECTOR_LINESTYLES[selector],
            color=SELECTOR_COLORS[selector],
            linewidth=1.3,
            markersize=5,
            label=SELECTOR_LABELS[selector],
        )
    for i, spec in enumerate(specs):
        top1 = next(row["top1_recovery"] for row in utility_spec_rows if row["utility_spec"] == spec and row["selector"] == "compiler")
        comp_regret = next(row["oracle_regret"] for row in utility_spec_rows if row["utility_spec"] == spec and row["selector"] == "compiler")
        ax_b.text(i, comp_regret + 0.0035, f"{100*top1:.0f}%", ha="center", va="bottom", fontsize=7.5, color=SELECTOR_COLORS["compiler"])
    ax_b.axhline(0, color="#777777", linewidth=0.8)
    ax_b.set_xticks(spec_x, [UTILITY_SPECS[spec] for spec in specs])
    ax_b.set_ylabel("Oracle regret")
    ax_b.set_title("Selection under utility specifications", loc="left")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(w_pad=2.1)

    dirs = ensure_dirs(output_root)
    plotted = [*summary_rows, *utility_spec_rows]
    write_summary(dirs["data"] / "exp3_selection_summary.csv", plotted)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "n_test_episodes": int(selection_rows["episode_id"].nunique()),
            "balanced_selector_summary": summary_rows,
            "utility_spec_summary": utility_spec_rows,
        }
    )
    save_outputs(fig, dirs["main"] / "fig_exp3_selection", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Main Figure 3: Experiment 3 program selection.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--experiment3_root", default="artifacts/llama/selection")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    geometry_path = Path(args.geometry)
    exp3 = Path(args.experiment3_root)
    compiler_csv = exp3 / "compiler_evaluation.csv"
    geometry, selection_rows = build_balanced_selection(geometry_path, compiler_csv)
    utility_rows = build_utility_spec_summary(geometry, exp3)
    inputs = [geometry_path, compiler_csv, *[utility_spec_path(exp3, spec, "csv") for spec in UTILITY_SPECS]]
    metadata = draw_figure(selection_rows, utility_rows, Path(args.output_root), inputs)
    balanced = {row["selector"]: row for row in metadata["balanced_selector_summary"]}
    print(
        "Main Fig. 3:",
        f"compiler_utility={format_number(balanced['compiler']['mean_utility'])}",
        f"compiler_regret={format_number(balanced['compiler']['oracle_regret'])}",
        f"oracle={format_number(balanced['oracle']['mean_utility'])}",
    )


if __name__ == "__main__":
    main()
