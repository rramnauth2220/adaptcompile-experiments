#!/usr/bin/env python3
"""Appendix Figure E: multi-seed robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    PROGRAM_COLORS,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    ensure_dirs,
    json_script_metadata,
    panel_label,
    parse_bool,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    write_summary,
)


def build_rows(variation_path: Path, stability_path: Path):
    if not variation_path.exists() or not stability_path.exists():
        print(f"[warn] Missing multi-seed robustness inputs: {variation_path}, {stability_path}")
        return None, None
    variation = read_csv(variation_path)
    stability = read_csv(stability_path)
    required_var = {"episode_id", "learning_type", "config_id", "std"}
    required_stab = {"episode_id", "learning_type", "common_oracle_exists", "same_unique_winner_all_seeds"}
    if missing := required_var - set(variation.columns):
        raise ValueError(f"Seed variation missing columns: {sorted(missing)}")
    if missing := required_stab - set(stability.columns):
        raise ValueError(f"Winner stability missing columns: {sorted(missing)}")
    summary = []
    for cfg in PROGRAM_ORDER:
        vals = variation.loc[variation["config_id"] == cfg, "std"].astype(float)
        summary.append(
            {
                "panel": "seed_variation",
                "config_id": cfg,
                "program": PROGRAM_LABELS[cfg],
                "n_episode_config_units": int(len(vals)),
                "mean_utility_sd": float(vals.mean()),
                "median_utility_sd": float(vals.median()),
            }
        )
    for lt in OBJECTIVE_ORDER:
        sub = stability[stability["learning_type"] == lt]
        if sub.empty:
            continue
        summary.append(
            {
                "panel": "winner_stability",
                "learning_type": lt,
                "objective": OBJECTIVE_LABELS[lt],
                "n_episodes": int(len(sub)),
                "strict_same_unique_winner_rate": float(sub["same_unique_winner_all_seeds"].map(parse_bool).mean()),
                "tie_aware_common_oracle_rate": float(sub["common_oracle_exists"].map(parse_bool).mean()),
            }
        )
    return variation, stability, summary


def draw_figure(variation, stability, summary, output_root: Path, inputs):
    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.85), gridspec_kw={"width_ratios": [1.0, 1.15]})

    x = np.arange(len(PROGRAM_ORDER))
    rng = np.random.default_rng(2026)
    for i, cfg in enumerate(PROGRAM_ORDER):
        vals = variation.loc[variation["config_id"] == cfg, "std"].astype(float).to_numpy()
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        ax_a.scatter(np.full(len(vals), i) + jitter, vals, s=14, color=PROGRAM_COLORS[cfg], alpha=0.55, linewidths=0)
        ax_a.scatter(i, np.median(vals), s=42, color="#000000", marker="_", linewidths=1.6, zorder=5)
    ax_a.set_xticks(x, [PROGRAM_LABELS[cfg] for cfg in PROGRAM_ORDER])
    ax_a.set_ylabel("Utility SD across seeds")
    ax_a.set_title("Seed variation by program", loc="left")
    prettify_axes(ax_a)
    panel_label(ax_a, "A")

    objectives = [lt for lt in OBJECTIVE_ORDER if lt in set(stability["learning_type"])]
    ox = np.arange(len(objectives))
    width = 0.32
    strict = []
    tie = []
    for lt in objectives:
        sub = stability[stability["learning_type"] == lt]
        strict.append(float(sub["same_unique_winner_all_seeds"].map(parse_bool).mean()))
        tie.append(float(sub["common_oracle_exists"].map(parse_bool).mean()))
    ax_b.barh(ox - width / 2, strict, height=width, color="#777777", alpha=0.75, label="Same unique winner")
    ax_b.barh(ox + width / 2, tie, height=width, color="#AA3377", alpha=0.85, label="Common oracle set")
    ax_b.set_yticks(ox, [OBJECTIVE_LABELS[lt] for lt in objectives])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.03)
    ax_b.xaxis.set_major_formatter(lambda value, _pos: f"{int(round(100*value))}%")
    ax_b.set_xlabel("Episode fraction")
    ax_b.set_title("Winner stability by objective", loc="left")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    prettify_axes(ax_b)
    panel_label(ax_b, "B")
    fig.tight_layout(w_pad=2.0)

    dirs = ensure_dirs(output_root)
    write_summary(dirs["data"] / "seed_robustness_summary.csv", summary)
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "summary": summary,
            "definitions": {
                "strict_same_unique_winner_rate": "Fraction of episodes with the same unique oracle winner across seeds.",
                "tie_aware_common_oracle_rate": "Fraction of episodes whose seed-specific oracle sets have a nonempty intersection.",
            },
        }
    )
    save_outputs(fig, dirs["appendix"] / "fig_seed_robustness", metadata)
    plt.close(fig)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot appendix multi-seed robustness figure.")
    parser.add_argument("--analysis_root", default="artifacts/llama/geometry/robustness")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()
    analysis = Path(args.analysis_root)
    variation_path = analysis / "per_config_seed_variation.csv"
    stability_path = analysis / "winner_stability.csv"
    built = build_rows(variation_path, stability_path)
    if built[0] is None:
        return
    variation, stability, summary = built
    draw_figure(variation, stability, summary, Path(args.output_root), [variation_path, stability_path])
    print("Appendix seed robustness:", f"episodes={stability['episode_id'].nunique()}")


if __name__ == "__main__":
    main()
