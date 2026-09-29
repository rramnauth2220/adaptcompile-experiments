#!/usr/bin/env python3
"""Appendix Figure A1: Experiment 1 oracle winner margins.

The figure uses the seed-aggregated TEST geometry and recomputes the
episode-level margin directly from the four primary budget-matched programs:

    margin = max_g U_D(g) - second_best_g U_D(g)

where U is balanced utility over acquisition, transfer, boundedness, and
preservation. The saved headroom margins CSV is used as an external sanity
check, not as a substitute for episode-level plotting data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from plot_style import (
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    PROGRAM_ORDER,
    SEED,
    assert_balanced_utility,
    assert_test_episode_grid,
    bootstrap_ci,
    deterministic_jitter,
    ensure_dirs,
    json_script_metadata,
    oracle_sets,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


def load_test_geometry(geometry_path: Path):
    geometry = read_csv(geometry_path)
    assert_balanced_utility(geometry)
    assert_test_episode_grid(geometry, configs=PROGRAM_ORDER)
    test = geometry[geometry["meta_split"] == "test"].copy()
    missing = [cfg for cfg in PROGRAM_ORDER if cfg not in set(test["config_id"])]
    if missing:
        raise ValueError(f"Missing primary configs in test geometry: {missing}")
    return test[test["config_id"].isin(PROGRAM_ORDER)].copy()


def summarize_margins(margins) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = [("overall", "all", margins)] + [
        ("learning_type", lt, margins[margins["learning_type"] == lt]) for lt in OBJECTIVE_ORDER
    ]
    for i, (group_type, group_value, sub) in enumerate(groups):
        vals = sub["top2_margin"].astype(float).to_numpy()
        mean, lo, hi = bootstrap_ci(vals, seed=SEED + i)
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "learning_type": "" if group_type == "overall" else group_value,
                "objective": "Overall" if group_type == "overall" else OBJECTIVE_LABELS[group_value],
                "n_episodes": int(sub["episode_id"].nunique()),
                "top2_margin_mean": mean,
                "top2_margin_ci_low": lo,
                "top2_margin_ci_high": hi,
                "top2_margin_median": float(np.median(vals)),
                "top2_margin_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "top2_margin_min": float(np.min(vals)),
                "top2_margin_max": float(np.max(vals)),
                "tie_rate": float((sub["oracle_set_size"].astype(int) > 1).mean()),
                "unique_winner_rate": float((sub["oracle_set_size"].astype(int) == 1).mean()),
            }
        )
    return rows


def compare_to_saved_headroom(summary_rows: List[Dict[str, Any]], margins_csv: Path | None) -> List[Dict[str, Any]]:
    if margins_csv is None or not margins_csv.exists():
        return []
    saved = read_csv(margins_csv)
    comparisons: List[Dict[str, Any]] = []
    summary_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row["group_type"]), str(row["group_value"])): row for row in summary_rows
    }
    for _, saved_row in saved.iterrows():
        group_type = str(saved_row["group_type"])
        group_value = str(saved_row["group_value"])
        key = (group_type, group_value)
        ours = summary_by_key.get(key)
        if ours is None:
            continue
        comparisons.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "mean_difference": float(ours["top2_margin_mean"] - float(saved_row["top_2_margin_mean"])),
                "median_difference": float(ours["top2_margin_median"] - float(saved_row["top_2_margin_median"])),
                "tie_rate_difference": float(ours["tie_rate"] - float(saved_row["top_tie_rate"])),
                "unique_winner_rate_difference": float(
                    ours["unique_winner_rate"] - float(saved_row["unique_winner_rate"])
                ),
            }
        )
    return comparisons


def build_rows(geometry_path: Path, margins_csv: Path | None = None):
    test = load_test_geometry(geometry_path)
    margins = oracle_sets(test)
    if margins["episode_id"].nunique() != 100:
        raise ValueError(f"Expected 100 test episode margins, found {margins['episode_id'].nunique()}.")
    counts = margins.groupby("learning_type")["episode_id"].nunique().to_dict()
    bad_counts = {lt: counts.get(lt, 0) for lt in OBJECTIVE_ORDER if counts.get(lt, 0) != 20}
    if bad_counts:
        raise ValueError(f"Expected 20 test episodes per objective, got {bad_counts}")
    summary_rows = summarize_margins(margins)
    comparisons = compare_to_saved_headroom(summary_rows, margins_csv)
    return margins, summary_rows, comparisons


def draw_figure(margins, summary_rows, comparisons, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(4.25, 2.85))

    y = np.arange(len(OBJECTIVE_ORDER), dtype=float)
    for i, lt in enumerate(OBJECTIVE_ORDER):
        sub = margins[margins["learning_type"] == lt]
        vals = sub["top2_margin"].astype(float).to_numpy()
        jitter = deterministic_jitter(len(vals), width=0.13, seed=SEED + i)
        ax.scatter(
            vals,
            np.full(len(vals), i) + jitter,
            s=17,
            color="#4D4D4D",
            alpha=0.52,
            linewidths=0,
            zorder=2,
        )

        row = next(r for r in summary_rows if r["learning_type"] == lt)
        mean = float(row["top2_margin_mean"])
        lo = float(row["top2_margin_ci_low"])
        hi = float(row["top2_margin_ci_high"])
        ax.errorbar(
            mean,
            i,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            markersize=5.0,
            color="#000000",
            ecolor="#000000",
            elinewidth=1.2,
            capsize=3.0,
            zorder=5,
        )

    ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    ax.set_yticks(y, [OBJECTIVE_LABELS[lt] for lt in OBJECTIVE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Oracle winner margin")
    ax.set_title("Oracle winner margins", loc="left")
    ax.set_xlim(left=-0.005, right=max(0.19, float(margins["top2_margin"].max()) + 0.01))
    ax.set_xticks(np.arange(0.0, 0.20, 0.05))
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.2f}")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()

    dirs = ensure_dirs(output_root)
    summary_path = dirs["data"] / "exp1_margins_summary.csv"
    write_summary(summary_path, summary_rows)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "A1 oracle winner margins",
            "definition": "best balanced utility minus second-best balanced utility over the four primary configs",
            "utility": "mean(acquisition, transfer, boundedness, preservation)",
            "n_bootstrap": N_BOOT,
            "n_test_episodes": int(margins["episode_id"].nunique()),
            "summary_csv": str(summary_path),
            "summary": summary_rows,
            "saved_headroom_comparison": comparisons,
        }
    )
    outputs = save_outputs(fig, dirs["appendix"] / "fig_exp1_margins", metadata)
    plt.close(fig)
    return outputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 1 appendix oracle winner margins.")
    parser.add_argument("--geometry", default="artifacts/llama/geometry/geometry_dataset.csv")
    parser.add_argument(
        "--margins_csv",
        default="artifacts/llama/geometry/headroom/headroom_top2_margins.csv",
    )
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    geometry_path = Path(args.geometry)
    margins_path = Path(args.margins_csv) if args.margins_csv else None
    margins, summary_rows, comparisons = build_rows(geometry_path, margins_path)
    inputs = [geometry_path] + ([margins_path] if margins_path is not None else [])
    draw_figure(margins, summary_rows, comparisons, Path(args.output_root), inputs)
    overall = next(row for row in summary_rows if row["group_type"] == "overall")
    print(
        "Appendix Exp1 margins:",
        f"episodes={overall['n_episodes']}",
        f"mean={overall['top2_margin_mean']:.5f}",
        f"median={overall['top2_margin_median']:.5f}",
        f"tie_rate={overall['tie_rate']:.3f}",
    )


if __name__ == "__main__":
    main()
