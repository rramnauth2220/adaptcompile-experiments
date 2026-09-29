#!/usr/bin/env python3
"""Appendix Figure A2: adaptation seed robustness for Experiment 1.

Panel A uses the seed-level robustness subset to summarize utility variation for
each episode x program cell. Panel B uses the tie-aware winner-stability audit:

* strict stability = same unique oracle winner for all three seeds
* tie-aware stability = seed-specific oracle sets have a nonempty intersection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from plot_style import (
    N_BOOT,
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    PROGRAM_COLORS,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    SEED,
    bootstrap_ci,
    ensure_dirs,
    json_script_metadata,
    parse_bool,
    prettify_axes,
    read_csv,
    save_outputs,
    setup_matplotlib,
    weighted_utility,
    write_summary,
)


EXPECTED_SEEDS = [11, 22, 33]


def _column_as_bool(series: pd.Series) -> pd.Series:
    return series.map(parse_bool).astype(bool)


def validate_raw_seed_records(raw_records: pd.DataFrame) -> Dict[str, Any]:
    required = {
        "episode_id",
        "learning_type",
        "seed",
        "config_id",
        "acquisition",
        "transfer",
        "boundedness",
        "preservation",
    }
    missing = required - set(raw_records.columns)
    if missing:
        raise ValueError(f"Seed-level robustness records missing columns: {sorted(missing)}")

    work = raw_records[raw_records["config_id"].isin(PROGRAM_ORDER)].copy()
    work["seed"] = work["seed"].astype(int)
    observed_seeds = sorted(work["seed"].unique().tolist())
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {observed_seeds}.")
    n_episodes = work["episode_id"].nunique()
    if n_episodes != 20:
        raise ValueError(f"Expected 20 multi-seed robustness episodes, found {n_episodes}.")
    duplicate_count = int(work.duplicated(["episode_id", "config_id", "seed"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate episode/config/seed records in robustness data: {duplicate_count}")

    expected = {
        (episode, cfg, seed)
        for episode in work["episode_id"].unique()
        for cfg in PROGRAM_ORDER
        for seed in EXPECTED_SEEDS
    }
    observed = set(zip(work["episode_id"], work["config_id"], work["seed"]))
    missing_cells = sorted(expected - observed)
    if missing_cells:
        raise ValueError(f"Missing robustness episode/config/seed cells; first missing cell: {missing_cells[0]}")

    if "exploratory_balanced_utility" in work.columns:
        diff = (work["exploratory_balanced_utility"].astype(float) - weighted_utility(work)).abs().max()
        if diff > 1e-9:
            raise ValueError(f"Seed-level balanced utility mismatch: max absolute difference {diff}")

    return {
        "n_episodes": int(n_episodes),
        "n_configs": int(work["config_id"].nunique()),
        "seeds": observed_seeds,
        "n_seed_level_rows": int(len(work)),
    }


def load_inputs(raw_records_path: Path, analysis_root: Path):
    variation_path = analysis_root / "per_config_seed_variation.csv"
    stability_path = analysis_root / "winner_stability.csv"
    robust_headroom_path = analysis_root / "robust_headroom_by_episode.csv"
    if not raw_records_path.exists():
        print(f"[warn] Missing raw multi-seed robustness records: {raw_records_path}")
        return None
    if not variation_path.exists() or not stability_path.exists():
        print(f"[warn] Missing multi-seed robustness analysis files under {analysis_root}")
        return None

    raw = read_csv(raw_records_path)
    raw_meta = validate_raw_seed_records(raw)
    variation = read_csv(variation_path)
    stability = read_csv(stability_path)
    robust_headroom = read_csv(robust_headroom_path) if robust_headroom_path.exists() else pd.DataFrame()

    required_var = {"episode_id", "learning_type", "config_id", "std", "range"}
    required_stab = {"episode_id", "learning_type", "common_oracle_exists", "same_unique_winner_all_seeds"}
    if missing := required_var - set(variation.columns):
        raise ValueError(f"Seed variation file missing columns: {sorted(missing)}")
    if missing := required_stab - set(stability.columns):
        raise ValueError(f"Winner stability file missing columns: {sorted(missing)}")

    variation = variation[variation["config_id"].isin(PROGRAM_ORDER)].copy()
    if variation["episode_id"].nunique() != 20:
        raise ValueError(f"Expected 20 episodes in seed-variation file, found {variation['episode_id'].nunique()}.")
    if len(variation) != 20 * len(PROGRAM_ORDER):
        raise ValueError(f"Expected 80 episode/config variation rows, found {len(variation)}.")
    if stability["episode_id"].nunique() != 20:
        raise ValueError(f"Expected 20 episodes in winner-stability file, found {stability['episode_id'].nunique()}.")

    return {
        "raw": raw,
        "raw_meta": raw_meta,
        "variation": variation,
        "stability": stability,
        "robust_headroom": robust_headroom,
        "paths": {
            "raw_records": raw_records_path,
            "variation": variation_path,
            "stability": stability_path,
            "robust_headroom": robust_headroom_path,
        },
    }


def build_summary(variation, stability, robust_headroom, raw_meta) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_std = variation["std"].astype(float).to_numpy()
    all_range = variation["range"].astype(float).to_numpy()
    rows.append(
        {
            "section": "overall_seed_variation",
            "n_episodes": raw_meta["n_episodes"],
            "n_configs": raw_meta["n_configs"],
            "n_episode_config_units": int(len(variation)),
            "seeds": "|".join(str(s) for s in raw_meta["seeds"]),
            "mean_utility_sd": float(np.mean(all_std)),
            "median_utility_sd": float(np.median(all_std)),
            "mean_utility_range": float(np.mean(all_range)),
        }
    )

    for cfg in PROGRAM_ORDER:
        sub = variation[variation["config_id"] == cfg]
        vals = sub["std"].astype(float).to_numpy()
        mean, lo, hi = bootstrap_ci(vals, seed=SEED + PROGRAM_ORDER.index(cfg))
        rows.append(
            {
                "section": "seed_variation_by_program",
                "config_id": cfg,
                "program": PROGRAM_LABELS[cfg],
                "n_episode_config_units": int(len(sub)),
                "mean_utility_sd": mean,
                "mean_utility_sd_ci_low": lo,
                "mean_utility_sd_ci_high": hi,
                "median_utility_sd": float(np.median(vals)),
                "mean_utility_range": float(sub["range"].astype(float).mean()),
            }
        )

    strict_bool = _column_as_bool(stability["same_unique_winner_all_seeds"])
    tie_bool = _column_as_bool(stability["common_oracle_exists"])
    rows.append(
        {
            "section": "overall_winner_stability",
            "n_episodes": int(stability["episode_id"].nunique()),
            "same_unique_winner_count": int(strict_bool.sum()),
            "same_unique_winner_rate": float(strict_bool.mean()),
            "common_oracle_count": int(tie_bool.sum()),
            "common_oracle_rate": float(tie_bool.mean()),
        }
    )
    for lt in OBJECTIVE_ORDER:
        sub = stability[stability["learning_type"] == lt]
        if sub.empty:
            continue
        strict = _column_as_bool(sub["same_unique_winner_all_seeds"])
        tie = _column_as_bool(sub["common_oracle_exists"])
        rows.append(
            {
                "section": "winner_stability_by_objective",
                "learning_type": lt,
                "objective": OBJECTIVE_LABELS[lt],
                "n_episodes": int(sub["episode_id"].nunique()),
                "same_unique_winner_count": int(strict.sum()),
                "same_unique_winner_rate": float(strict.mean()),
                "common_oracle_count": int(tie.sum()),
                "common_oracle_rate": float(tie.mean()),
            }
        )

    if not robust_headroom.empty:
        rows.append(
            {
                "section": "seed_averaged_headroom",
                "n_episodes": int(robust_headroom["episode_id"].nunique()),
                "global_utility_mean": float(robust_headroom["global_utility"].astype(float).mean()),
                "objective_fixed_utility_mean": float(robust_headroom["objective_utility"].astype(float).mean()),
                "oracle_utility_mean": float(robust_headroom["oracle_utility"].astype(float).mean()),
                "objective_to_oracle_regret_mean": float(
                    robust_headroom["objective_to_oracle_regret"].astype(float).mean()
                ),
                "top2_margin_mean": float(robust_headroom["top2_margin"].astype(float).mean()),
                "top2_margin_median": float(robust_headroom["top2_margin"].astype(float).median()),
            }
        )
    return rows


def draw_figure(payload, output_root: Path):
    variation = payload["variation"]
    stability = payload["stability"]
    summary_rows = build_summary(variation, stability, payload["robust_headroom"], payload["raw_meta"])

    plt = setup_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 3.05), gridspec_kw={"width_ratios": [1.0, 1.18]})

    for i, cfg in enumerate(PROGRAM_ORDER):
        sub = variation[variation["config_id"] == cfg]
        vals = sub["std"].astype(float).to_numpy()
        jitter = np.random.default_rng(SEED + i).uniform(-0.12, 0.12, size=len(vals))
        ax_a.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=16,
            color=PROGRAM_COLORS[cfg],
            alpha=0.62,
            linewidths=0,
            label=PROGRAM_LABELS[cfg],
        )
        row = next(
            r
            for r in summary_rows
            if r.get("section") == "seed_variation_by_program" and r.get("config_id") == cfg
        )
        mean = float(row["mean_utility_sd"])
        lo = float(row["mean_utility_sd_ci_low"])
        hi = float(row["mean_utility_sd_ci_high"])
        ax_a.errorbar(
            i,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            markersize=4.8,
            color="#000000",
            ecolor="#000000",
            elinewidth=1.1,
            capsize=3,
            zorder=5,
        )
    ax_a.set_xticks(np.arange(len(PROGRAM_ORDER)), [PROGRAM_LABELS[cfg] for cfg in PROGRAM_ORDER])
    ax_a.set_ylabel("Utility SD across seeds")
    ax_a.set_title("Seed variation", loc="left")
    ax_a.set_ylim(bottom=-0.002)
    prettify_axes(ax_a)

    objectives = [lt for lt in OBJECTIVE_ORDER if lt in set(stability["learning_type"])]
    positions = np.arange(len(objectives))
    strict_vals = []
    tie_vals = []
    for lt in objectives:
        sub = stability[stability["learning_type"] == lt]
        strict_vals.append(float(_column_as_bool(sub["same_unique_winner_all_seeds"]).mean()))
        tie_vals.append(float(_column_as_bool(sub["common_oracle_exists"]).mean()))
    width = 0.28
    ax_b.barh(
        positions - width / 2,
        strict_vals,
        height=width,
        color="#777777",
        alpha=0.78,
        label="Same unique winner",
    )
    ax_b.barh(
        positions + width / 2,
        tie_vals,
        height=width,
        color="#AA3377",
        alpha=0.86,
        label="Common oracle program",
    )
    ax_b.set_yticks(positions, [OBJECTIVE_LABELS[lt] for lt in objectives])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.04)
    ax_b.xaxis.set_major_formatter(lambda value, _pos: f"{int(round(100 * value))}%")
    ax_b.set_xlabel("Episode fraction")
    ax_b.set_title("Winner stability", loc="left")
    ax_b.grid(axis="x")
    ax_b.set_axisbelow(True)

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)

    dirs = ensure_dirs(output_root)
    summary_path = dirs["data"] / "exp1_seed_robustness_summary.csv"
    write_summary(summary_path, summary_rows)
    inputs = list(payload["paths"].values())
    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "figure": "A2 adaptation seed robustness",
            "summary_csv": str(summary_path),
            "definitions": {
                "utility_sd": "standard deviation of balanced utility across seeds within an episode x program cell",
                "strict_stability": "fraction of episodes with the same unique oracle winner for all seeds",
                "tie_aware_stability": "fraction of episodes whose per-seed oracle sets have a nonempty intersection",
            },
            "raw_integrity": payload["raw_meta"],
            "summary": summary_rows,
            "bootstrap_resamples": N_BOOT,
        }
    )
    outputs = save_outputs(fig, dirs["appendix"] / "fig_seed_robustness", metadata)
    plt.close(fig)
    return outputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 1 appendix seed robustness.")
    parser.add_argument("--raw_records", default="artifacts/llama/geometry/robustness/geometry_records_3seed.csv")
    parser.add_argument("--analysis_root", default="artifacts/llama/geometry/robustness")
    parser.add_argument("--output_root", default="outputs/release_verification/figures/llama")
    args = parser.parse_args()

    payload = load_inputs(Path(args.raw_records), Path(args.analysis_root))
    if payload is None:
        return
    _, metadata = draw_figure(payload, Path(args.output_root))
    overall = next(row for row in metadata["summary"] if row["section"] == "overall_seed_variation")
    stability = next(row for row in metadata["summary"] if row["section"] == "overall_winner_stability")
    print(
        "Appendix Exp1 seed robustness:",
        f"episodes={overall['n_episodes']}",
        f"mean_sd={overall['mean_utility_sd']:.5f}",
        f"median_sd={overall['median_utility_sd']:.5f}",
        f"strict={stability['same_unique_winner_count']}/{stability['n_episodes']}",
        f"tie_aware={stability['common_oracle_count']}/{stability['n_episodes']}",
    )


if __name__ == "__main__":
    main()
