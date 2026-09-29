#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci(values, n_boot=10000, ci=95, seed=2026):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    if len(values) == 1:
        return values[0], values[0]

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2
    lo = np.percentile(boot_means, alpha)
    hi = np.percentile(boot_means, 100 - alpha)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="outputs/llama31_8b_behavioral_multiseed/summary_behavioral_policy_multiseed_per_seed_compact.csv",
    )
    parser.add_argument(
        "--output_prefix",
        default="outputs/llama31_8b_behavioral_multiseed/behavioral_policy_multiseed_ci_truncated",
    )
    parser.add_argument("--selected_budget", type=int, default=10)
    parser.add_argument("--ci", type=float, default=95)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--y_min", type=float, default=0.55)
    parser.add_argument("--y_max", type=float, default=1.02)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required = {"budget", "seed", "acquisition", "generalization", "boundedness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metrics = [
        ("acquisition", "Acquisition"),
        ("generalization", "Generalization"),
        ("boundedness", "Boundedness"),
    ]

    rows = []
    for budget in sorted(df["budget"].unique()):
        sub = df[df["budget"] == budget]

        for col, label in metrics:
            vals = sub[col].dropna().to_numpy(dtype=float)
            mean = vals.mean()
            lo, hi = bootstrap_ci(
                vals,
                n_boot=args.n_boot,
                ci=args.ci,
                seed=2026 + int(budget),
            )

            rows.append(
                {
                    "budget": budget,
                    "metric": col,
                    "metric_label": label,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": len(vals),
                }
            )

    agg = pd.DataFrame(rows)

    # Relative plateau thresholds, consistent with lexical/factual calibration.
    acq_best = agg.loc[agg["metric"] == "acquisition", "mean"].max()
    transfer_best = agg.loc[agg["metric"] == "generalization", "mean"].max()

    acq_threshold = 0.95 * acq_best
    transfer_threshold = 0.95 * transfer_best

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    agg.to_csv(out_prefix.with_suffix(".csv"), index=False)

    plt.figure(figsize=(7.4, 4.8))

    for col, label in metrics:
        g = agg[agg["metric"] == col].sort_values("budget")

        yerr = np.vstack(
            [
                g["mean"].to_numpy() - g["ci_low"].to_numpy(),
                g["ci_high"].to_numpy() - g["mean"].to_numpy(),
            ]
        )

        plt.errorbar(
            g["budget"],
            g["mean"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
        )

    # Budget-selection thresholds.
    plt.axhline(acq_threshold, linestyle=":", linewidth=1)
    plt.axhline(transfer_threshold, linestyle="--", linewidth=1)

    x_left = sorted(df["budget"].unique())[0]
    plt.text(
        x_left,
        acq_threshold + 0.006,
        f"95% acquisition threshold = {acq_threshold:.3f}",
        fontsize=9,
        va="bottom",
    )
    plt.text(
        x_left,
        transfer_threshold - 0.010,
        f"95% transfer threshold = {transfer_threshold:.3f}",
        fontsize=9,
        va="top",
    )

    # Selected budget marker.
    plt.axvline(args.selected_budget, linestyle="--", linewidth=1)
    plt.text(
        args.selected_budget + 0.04,
        args.y_min + 0.02,
        "selected\nbudget",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=9,
    )

    plt.xlabel("Training examples per specification")
    plt.ylabel("Accuracy")
    plt.title("Behavioral policy multi-seed calibration")
    plt.xticks(sorted(df["budget"].unique()))
    plt.ylim(args.y_min, args.y_max)

    plt.legend(loc="lower right", frameon=True)

    plt.figtext(
        0.5,
        0.005,
        "Mean ± 95% bootstrap CI across 3 seeds. "
        "Boundedness = NO_POLICY_TRIGGER accuracy on negative controls. "
        "Y-axis truncated to show trends.",
        ha="center",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print("\nBootstrap summary:")
    print(agg.round(3).to_string(index=False))

    print("\nThresholds:")
    print(f"95% acquisition threshold: {acq_threshold:.3f}")
    print(f"95% transfer threshold:    {transfer_threshold:.3f}")

    print("\nWrote:")
    print(png_path)
    print(pdf_path)
    print(out_prefix.with_suffix(".csv"))


if __name__ == "__main__":
    main()
