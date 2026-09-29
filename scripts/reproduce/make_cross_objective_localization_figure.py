#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_STYLES = {
    "full":   {"label": "Full",   "color": "#CC79A7"},  # distinct magenta
    "early":  {"label": "Early",  "color": "#D0DCEB"},  # very light blue-gray
    "middle": {"label": "Middle", "color": "#92A9C1"},  # medium blue-gray
    "late":   {"label": "Late",   "color": "#5A7693"},  # dark blue-gray
}

OBJECTIVES = [
    {
        "key": "lexical_binding",
        "label": "Lexical",
        "budget": 10,
        "match": "lexical",
        "boundedness_source": "strict",
    },
    {
        "key": "factual_association",
        "label": "Factual",
        "budget": 8,
        "match": "factual",
        "boundedness_source": "strict",
    },
    {
        "key": "behavioral_policy",
        "label": "Behavioral",
        "budget": 10,
        "match": "behavioral",
        "boundedness_source": "concept",
    },
    {
        "key": "causal_mapping",
        "label": "Causal",
        "budget": 10,
        "match": "causal",
        "boundedness_source": "concept",
    },
    {
        "key": "procedural_reasoning",
        "label": "Procedural",
        "budget": 8,
        "match": "procedural",
        "boundedness_source": "concept",
    },
]

CONDITION_ORDER = ["full", "early", "middle", "late"]
CONDITION_LABELS = {
    "full": "Full",
    "early": "Early",
    "middle": "Middle",
    "late": "Late",
}

METRICS = [
    ("acquisition", "A. Acquisition"),
    ("generalization", "B. Transfer / generalization"),
    ("boundedness", "C. Boundedness"),
]


def load_release_by_seed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(path)
    required = {
        "seed", "localization_condition", "id_eval", "paraphrase_eval",
        "acquisition", "generalization", "boundedness", "objective_key",
        "objective_label", "budget", "source_dir",
    }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out["localization_condition"] = out["localization_condition"].str.lower()
    return out[out["localization_condition"].isin(CONDITION_ORDER)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the cross-objective localization figure from compact release evidence.")
    parser.add_argument("--input", default="artifacts/localization/primary/cross_objective_localization/cross_objective_localization_profile_by_seed.csv")
    parser.add_argument("--output-dir", default="outputs/release_verification/figures/localization")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed = load_release_by_seed(Path(args.input))

    objective_order = [o["key"] for o in OBJECTIVES]
    per_seed["objective_key"] = pd.Categorical(
        per_seed["objective_key"], objective_order, ordered=True
    )
    per_seed["localization_condition"] = pd.Categorical(
        per_seed["localization_condition"], CONDITION_ORDER, ordered=True
    )
    per_seed = per_seed.sort_values(["objective_key", "localization_condition", "seed"])

    per_seed_path = outdir / "cross_objective_localization_profile_by_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    agg = (
        per_seed.groupby(
            ["objective_key", "objective_label", "budget", "localization_condition"],
            observed=True,
            as_index=False,
        )
        .agg(
            acquisition=("acquisition", "mean"),
            acquisition_sd=("acquisition", "std"),
            generalization=("generalization", "mean"),
            generalization_sd=("generalization", "std"),
            boundedness=("boundedness", "mean"),
            boundedness_sd=("boundedness", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    # Fail if any condition is missing.
    expected_rows = len(OBJECTIVES) * len(CONDITION_ORDER)
    if len(agg) != expected_rows:
        print(agg)
        raise ValueError(f"Expected {expected_rows} rows, found {len(agg)}.")

    # SD is allowed to be exactly zero, but should not be missing with 3 seeds.
    sd_cols = ["acquisition_sd", "generalization_sd", "boundedness_sd"]
    if agg[sd_cols].isna().any().any():
        print(agg[agg[sd_cols].isna().any(axis=1)])
        raise ValueError("Some SD values are missing. Check per-seed inputs.")

    agg_path = outdir / "cross_objective_localization_profile.csv"
    agg.to_csv(agg_path, index=False)

    x = np.arange(len(OBJECTIVES))
    width = 0.18
    jitter_offsets = np.array([-0.035, 0.0, 0.035])

    objective_labels = [f"{o['label']}\nB={o['budget']}" for o in OBJECTIVES]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9), sharey=True)

    for ax, (metric, title) in zip(axes, METRICS):
        for i, condition in enumerate(CONDITION_ORDER):
            centers = x + (i - 1.5) * width
            means = []
            sds = []

            for obj in OBJECTIVES:
                sub = agg[
                    (agg["objective_key"].astype(str) == obj["key"])
                    & (agg["localization_condition"].astype(str) == condition)
                ]

                means.append(float(sub[metric].iloc[0]))
                sds.append(float(sub[f"{metric}_sd"].iloc[0]))

            style = CONDITION_STYLES[condition]

            ax.bar(
                centers,
                means,
                width=width,
                yerr=sds,
                capsize=3,
                linewidth=0.6,
                edgecolor="black",
                color=style["color"],
                label=style["label"],
                alpha=0.9,
            )

            # Overlay individual seeds.
            for j, obj in enumerate(OBJECTIVES):
                seed_sub = per_seed[
                    (per_seed["objective_key"].astype(str) == obj["key"])
                    & (per_seed["localization_condition"].astype(str) == condition)
                ].sort_values("seed")

                seed_vals = seed_sub[metric].to_numpy(dtype=float)
                n = len(seed_vals)

                if n == 1:
                    xs = np.array([centers[j]])
                elif n == 2:
                    xs = centers[j] + np.array([-0.025, 0.025])
                elif n == 3:
                    xs = centers[j] + jitter_offsets
                else:
                    xs = centers[j] + np.linspace(-0.04, 0.04, n)

                ax.scatter(
                    xs,
                    seed_vals,
                    s=12,
                    color="black",
                    alpha=0.5,
                    linewidths=0,
                    zorder=5,
                )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(objective_labels)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Learning objective")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Adaptation region",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
    )

    fig.suptitle(
        "Localized adaptation reveals distinct learning signatures",
        y=1.12,
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.025,
        "Bars show means across seeds; error bars show SD; black points show individual seed runs. "
        "Boundedness is objective-specific negative-control accuracy.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    png_path = outdir / "main_cross_objective_localization.png"
    pdf_path = outdir / "main_cross_objective_localization.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    figure_block = r"""\begin{figure*}[t]
    \centering
    \includegraphics[width=\linewidth]{main_cross_objective_localization.pdf}
    \caption{
    Cross-objective localization signatures across five calibrated learning objectives. Each objective is evaluated at its selected budget. Bars show mean performance across seeds, error bars show standard deviation across seeds, and black points show individual seed runs. Acquisition is the mean of in-distribution and paraphrase accuracy. Transfer is accuracy on held-out generalization examples. Boundedness is objective-specific negative-control accuracy. The figure shows that localized adaptation does not produce a single universal pattern: lexical binding is unusually early-localizable, factual association favors later adaptation among constrained adapters, behavioral policy is distributed, causal mapping is strongest in middle/full-stack adaptation, and procedural reasoning shows a tradeoff between middle-layer transfer and late-layer boundedness.
    }
    \label{fig:cross-objective-localization}
\end{figure*}
"""
    tex_path = outdir / "main_cross_objective_localization_figure_block.tex"
    tex_path.write_text(figure_block, encoding="utf-8")

    print("Wrote:")
    print(per_seed_path)
    print(agg_path)
    print(png_path)
    print(pdf_path)
    print(tex_path)

    print("\nCollected sources:")
    for src in per_seed["source_dir"].drop_duplicates():
        print(src)

    print("\nZero SD rows, if any:")
    zero = agg[
        (agg["acquisition_sd"] == 0)
        | (agg["generalization_sd"] == 0)
        | (agg["boundedness_sd"] == 0)
    ]
    if zero.empty:
        print("None")
    else:
        cols = [
            "objective_label",
            "localization_condition",
            "acquisition",
            "acquisition_sd",
            "generalization",
            "generalization_sd",
            "boundedness",
            "boundedness_sd",
            "n_seeds",
        ]
        print(zero[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
