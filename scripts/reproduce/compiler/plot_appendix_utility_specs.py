#!/usr/bin/env python3
"""Appendix Figure C: utility-specification robustness details."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    SELECTOR_COLORS,
    SELECTOR_LABELS,
    SELECTOR_LINESTYLES,
    SELECTOR_MARKERS,
    UTILITY_SPECS,
    assert_balanced_utility,
    assert_test_episode_grid,
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


def spec_file(root: Path, spec: str, suffix: str) -> Path:
    return root / f"{spec}.{suffix}"


def build_rows(geometry_path: Path, utility_root: Path):
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry)
    regret_rows = []
    switch_rows = []
    inputs = [geometry_path]
    balanced = read_csv(spec_file(utility_root, "balanced", "csv"))[["episode_id", "selected_config_id"]].rename(
        columns={"selected_config_id": "balanced_selected_config_id"}
    )
    inputs.append(spec_file(utility_root, "balanced", "csv"))
    for spec, label in UTILITY_SPECS.items():
        csv_path = spec_file(utility_root, spec, "csv")
        json_path = spec_file(utility_root, spec, "json")
        inputs.extend([csv_path, json_path])
        summary = read_json(json_path)
        detail = episode_selector_utilities(geometry, weights=summary["utility_weights"], compiler_csv=csv_path)
        regret_rows.extend(
            [
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "selector": "global_fixed",
                    "selector_label": SELECTOR_LABELS["global_fixed"],
                    "oracle_regret": float(detail["global_fixed_regret"].mean()),
                    "top1_recovery": np.nan,
                    "top2_recovery": np.nan,
                },
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "selector": "objective_fixed",
                    "selector_label": SELECTOR_LABELS["objective_fixed"],
                    "oracle_regret": float(detail["objective_fixed_regret"].mean()),
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
        if spec != "balanced":
            current = read_csv(csv_path)[["episode_id", "selected_config_id"]].rename(
                columns={"selected_config_id": "selected_config_id"}
            )
            merged = balanced.merge(current, on="episode_id", how="inner")
            changed = merged["balanced_selected_config_id"].astype(str) != merged["selected_config_id"].astype(str)
            switch_rows.append(
                {
                    "utility_spec": spec,
                    "utility_label": label,
                    "n_episodes": int(len(merged)),
                    "fraction_changed_vs_balanced": float(changed.mean()),
                }
            )
    return regret_rows, switch_rows, inputs


def draw_figure(regret_rows, switch_rows, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw={"width_ratios": [1.15, 0.85]})
    specs = list(UTILITY_SPECS)
    x = np.arange(len(specs))
    for selector in ["global_fixed", "objective_fixed", "compiler"]:
        vals = [
            next(row["oracle_regret"] for row in regret_rows if row["utility_spec"] == spec and row["selector"] == selector)
            for spec in specs
        ]
        ax_a.plot(
            x,
            vals,
            marker=SELECTOR_MARKERS[selector],
            linestyle=SELECTOR_LINESTYLES[selector],
            color=SELECTOR_COLORS[selector],
            linewidth=1.3,
            markersize=5,
            label=SELECTOR_LABELS[selector],
        )
    ax_a.set_xticks(x, [UTILITY_SPECS[spec] for spec in specs])
    ax_a.set_ylabel("Oracle regret")
    ax_a.set_title("Regret under utility specifications", loc="left")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    sx = np.arange(len(switch_rows))
    ax_b.bar(
        sx,
        [row["fraction_changed_vs_balanced"] for row in switch_rows],
        color=SELECTOR_COLORS["compiler"],
        alpha=0.82,
    )
    for i, row in enumerate(switch_rows):
        ax_b.text(i, row["fraction_changed_vs_balanced"] + 0.01, f"{100*row['fraction_changed_vs_balanced']:.0f}%", ha="center", fontsize=8)
    ax_b.set_xticks(sx, [row["utility_label"] for row in switch_rows])
    ax_b.set_ylim(0, max(0.2, max(row["fraction_changed_vs_balanced"] for row in switch_rows) + 0.06))
    ax_b.yaxis.set_major_formatter(lambda value, _pos: f"{int(round(100*value))}%")
    ax_b.set_ylabel("Episodes changed")
    ax_b.set_title("Compiler switching vs balanced", loc="left")
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(w_pad=2.0)

    dirs = ensure_dirs(output_root)
    write_summary(dirs["data"] / "utility_spec_summary.csv", [*regret_rows, *switch_rows])
    metadata = json_script_metadata(__file__, inputs)
    metadata.update({"regret_summary": regret_rows, "switching_summary": switch_rows})
    save_outputs(fig, dirs["appendix"] / "fig_utility_specifications", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appendix utility-specification figure.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--utility_root", default="artifacts/llama/selection/utility_specs")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    regret_rows, switch_rows, inputs = build_rows(Path(args.geometry), Path(args.utility_root))
    metadata = draw_figure(regret_rows, switch_rows, Path(args.output_root), inputs)
    balanced_compiler = next(row for row in metadata["regret_summary"] if row["utility_spec"] == "balanced" and row["selector"] == "compiler")
    print("Appendix utility specs:", f"balanced_compiler_regret={format_number(balanced_compiler['oracle_regret'])}")


if __name__ == "__main__":
    main()
