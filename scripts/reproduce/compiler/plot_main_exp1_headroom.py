#!/usr/bin/env python3
"""Main Figure 1: Experiment 1 selection headroom."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    PROGRAM_COLORS,
    PROGRAM_HATCHES,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    SEED,
    TIE_TOLERANCE,
    assert_balanced_utility,
    assert_test_episode_grid,
    bootstrap_ci,
    deterministic_jitter,
    ensure_dirs,
    episode_selector_utilities,
    fractional_oracle_winner_shares,
    format_number,
    json_script_metadata,
    objective_label,
    oracle_sets,
    panel_label,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


def build_summaries(geometry_path: Path) -> tuple:
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry)
    test = geometry[geometry["meta_split"] == "test"].copy()
    oracle = oracle_sets(test)
    shares = fractional_oracle_winner_shares(oracle)

    all_rows = []
    for cfg in PROGRAM_ORDER:
        credit = 0.0
        for _, row in oracle.iterrows():
            configs = [x for x in str(row["oracle_config_set"]).split("|") if x]
            if cfg in configs:
                credit += 1.0 / len(configs)
        all_rows.append({"learning_type": "all", "config_id": cfg, "share": credit / len(oracle)})
    utility_rows = episode_selector_utilities(geometry)
    headroom = utility_rows.rename(columns={"objective_fixed_regret": "objective_headroom"})
    headroom_summary = []
    for lt in OBJECTIVE_ORDER:
        vals = headroom.loc[headroom["learning_type"] == lt, "objective_headroom"].to_numpy(float)
        mean, lo, hi = bootstrap_ci(vals)
        headroom_summary.append(
            {
                "learning_type": lt,
                "objective": objective_label(lt),
                "n_episodes": len(vals),
                "mean_objective_to_oracle_headroom": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return shares, all_rows, headroom, headroom_summary, utility_rows


def draw_figure(shares, all_rows, headroom, headroom_summary, output_root: Path, geometry_path: Path) -> dict:
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(6.75, 2.95),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )

    panel_rows = [*OBJECTIVE_ORDER]
    summary_rows_order = [*OBJECTIVE_ORDER, "all"]
    y = np.arange(len(panel_rows))
    left = np.zeros(len(panel_rows))
    share_lookup = {}
    for _, row in shares.iterrows():
        share_lookup[(str(row["learning_type"]), str(row["config_id"]))] = float(row["share"])
    for row in all_rows:
        share_lookup[(row["learning_type"], row["config_id"])] = float(row["share"])

    legend_handles = []
    for cfg in PROGRAM_ORDER:
        vals = [share_lookup.get((lt, cfg), 0.0) for lt in panel_rows]
        bars = ax_a.barh(
            y,
            vals,
            left=left,
            color=PROGRAM_COLORS[cfg],
            edgecolor="white",
            linewidth=0.5,
            hatch=PROGRAM_HATCHES[cfg],
            label=PROGRAM_LABELS[cfg],
        )
        legend_handles.append(bars[0])
        left += np.asarray(vals)
    ax_a.set_yticks(y, [OBJECTIVE_LABELS[row] for row in panel_rows])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("Oracle-optimal share")
    ax_a.set_title("Oracle-optimal program", loc="left")
    ax_a.xaxis.set_major_formatter(lambda x, _pos: f"{int(round(100*x))}%")
    panel_label(ax_a, "A")

    for idx, lt in enumerate(OBJECTIVE_ORDER):
        subset = headroom[headroom["learning_type"] == lt].copy()
        vals = subset["objective_headroom"].to_numpy(float)
        jitter = deterministic_jitter(len(vals), width=0.13, seed=SEED + idx)
        ax_b.scatter(
            vals,
            np.full(len(vals), idx) + jitter,
            s=13,
            color="#555555",
            alpha=0.42,
            linewidths=0,
        )
        summary = next(row for row in headroom_summary if row["learning_type"] == lt)
        ax_b.errorbar(
            summary["mean_objective_to_oracle_headroom"],
            idx,
            xerr=[
                [summary["mean_objective_to_oracle_headroom"] - summary["ci95_low"]],
                [summary["ci95_high"] - summary["mean_objective_to_oracle_headroom"]],
            ],
            fmt="o",
            color="#000000",
            markersize=5.0,
            capsize=3,
            linewidth=1.1,
            zorder=5,
        )
    ax_b.axvline(0, color="#777777", linewidth=0.8)
    ax_b.tick_params(axis="y", labelleft=False, length=0)
    ax_b.set_xlabel("Episode-specific headroom")
    ax_b.set_title("Episode-specific headroom", loc="left")
    ax_b.set_xlim(left=-0.005, right=max(0.17, float(headroom["objective_headroom"].max()) + 0.01))
    ax_b.set_xticks(np.arange(0.0, 0.18, 0.05))
    ax_b.xaxis.set_major_formatter(lambda x, _pos: f"{x:.2f}")
    ax_b.grid(axis="x")
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "B")
    fig.legend(
        legend_handles,
        [PROGRAM_LABELS[cfg] for cfg in PROGRAM_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)

    dirs = ensure_dirs(output_root)
    summary_rows = []
    for lt in summary_rows_order:
        for cfg in PROGRAM_ORDER:
            summary_rows.append(
                {
                    "learning_type": lt,
                    "objective": OBJECTIVE_LABELS.get(lt, "All"),
                    "config_id": cfg,
                    "program": PROGRAM_LABELS[cfg],
                    "oracle_fractional_share": share_lookup.get((lt, cfg), 0.0),
                }
            )
    for row in headroom_summary:
        summary_rows.append(row)
    write_summary(dirs["data"] / "exp1_headroom_summary.csv", summary_rows)

    overall_headroom = float(headroom["objective_headroom"].mean())
    metadata = json_script_metadata(__file__, [geometry_path])
    metadata.update(
        {
            "n_test_episodes": int(headroom["episode_id"].nunique()),
            "objective_fixed_config_ids": dict(
                sorted(headroom.groupby("learning_type")["objective_fixed_config_id"].first().items())
            ),
            "global_fixed_config_id": str(headroom["global_fixed_config_id"].iloc[0]),
            "mean_objective_to_oracle_headroom": overall_headroom,
            "headroom_summary": headroom_summary,
            "oracle_share_all": {
                PROGRAM_LABELS[cfg]: share_lookup.get(("all", cfg), 0.0)
                for cfg in PROGRAM_ORDER
            },
        }
    )
    save_outputs(fig, dirs["main"] / "fig_exp1_headroom", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Main Figure 1: Experiment 1 selection headroom.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    geometry = Path(args.geometry)
    shares, all_rows, headroom, headroom_summary, _utility_rows = build_summaries(geometry)
    metadata = draw_figure(shares, all_rows, headroom, headroom_summary, Path(args.output_root), geometry)
    print(
        "Main Fig. 1:",
        f"n_test={metadata['n_test_episodes']}",
        f"mean_headroom={format_number(metadata['mean_objective_to_oracle_headroom'])}",
    )


if __name__ == "__main__":
    main()
