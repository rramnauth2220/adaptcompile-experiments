#!/usr/bin/env python3
"""
Analyze mislocation penalties from localization experiments.

The localization experiment already contains the relevant forced-mislocation
comparisons: for each objective, compare a preferred localized region against
a non-preferred/mislocated region.

Expected input columns:
  objective or objective_key or learning_type
  condition or localization_condition
  seed
  acquisition, transfer, boundedness

Also accepts:
  generalization instead of transfer
  id_eval + paraphrase_eval instead of acquisition

Outputs:
  mislocation_penalties_by_seed.csv
  mislocation_penalty_summary_long.csv
  mislocation_penalty_summary_wide.csv
  mislocation_penalty_table.tex
  mislocation_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PRIMARY_METRICS = ["acquisition", "transfer", "boundedness"]
DERIVED_METRICS = ["balanced_mean", "balanced_min"]
ALL_METRICS = PRIMARY_METRICS + DERIVED_METRICS

CONDITIONS = ["full", "early", "middle", "late"]


# These are the mislocation comparisons implied by the current results.
# Each row compares preferred_condition - mislocated_condition.
MISLOCATION_CONTRASTS: List[Dict[str, str]] = [
    {
        "objective": "lexical_binding",
        "contrast_id": "lexical_early_vs_late",
        "component": "lexical binding",
        "preferred_condition": "early",
        "mislocated_condition": "late",
        "interpretation": "late-layer adaptation is mislocated for lexical acquisition and bounded lexical binding",
    },
    {
        "objective": "lexical_binding",
        "contrast_id": "lexical_early_vs_middle",
        "component": "lexical binding",
        "preferred_condition": "early",
        "mislocated_condition": "middle",
        "interpretation": "middle-layer adaptation is less suited than early-layer adaptation for lexical binding",
    },
    {
        "objective": "factual_association",
        "contrast_id": "factual_late_vs_early",
        "component": "relational fact learning",
        "preferred_condition": "late",
        "mislocated_condition": "early",
        "interpretation": "early-layer adaptation is mislocated for factual association",
    },
    {
        "objective": "factual_association",
        "contrast_id": "factual_late_vs_middle",
        "component": "relational fact learning",
        "preferred_condition": "late",
        "mislocated_condition": "middle",
        "interpretation": "middle-layer adaptation is less suited than late-layer adaptation for factual association",
    },
    {
        "objective": "behavioral_policy",
        "contrast_id": "behavioral_gating_middle_vs_late",
        "component": "policy gating / boundedness",
        "preferred_condition": "middle",
        "mislocated_condition": "late",
        "interpretation": "late-layer adaptation preserves policy actions but is mislocated for bounded policy application",
    },
    {
        "objective": "behavioral_policy",
        "contrast_id": "behavioral_acquisition_late_vs_middle",
        "component": "policy acquisition",
        "preferred_condition": "late",
        "mislocated_condition": "middle",
        "interpretation": "middle-layer adaptation is less suited than late-layer adaptation for action-label acquisition",
    },
    {
        "objective": "causal_mapping",
        "contrast_id": "causal_middle_vs_early",
        "component": "causal transfer",
        "preferred_condition": "middle",
        "mislocated_condition": "early",
        "interpretation": "early-layer adaptation is mislocated for robust causal transfer",
    },
    {
        "objective": "causal_mapping",
        "contrast_id": "causal_middle_vs_late",
        "component": "causal transfer",
        "preferred_condition": "middle",
        "mislocated_condition": "late",
        "interpretation": "late-layer adaptation is less suited than middle-layer adaptation for causal transfer",
    },
    {
        "objective": "procedural_reasoning",
        "contrast_id": "procedural_middle_vs_early",
        "component": "procedural transfer",
        "preferred_condition": "middle",
        "mislocated_condition": "early",
        "interpretation": "early-layer adaptation is mislocated for procedural transfer",
    },
    {
        "objective": "procedural_reasoning",
        "contrast_id": "procedural_middle_vs_late",
        "component": "balanced procedural application",
        "preferred_condition": "middle",
        "mislocated_condition": "late",
        "interpretation": "late-layer adaptation is more conservative and less suited for balanced procedural application",
    },
]


OBJECTIVE_ALIASES = {
    "Lexical binding": "lexical_binding",
    "Factual association": "factual_association",
    "Behavioral policy": "behavioral_policy",
    "Causal mapping": "causal_mapping",
    "Procedural reasoning": "procedural_reasoning",
}


def normalize_objective_name(x: Any) -> str:
    s = str(x).strip()
    if s in OBJECTIVE_ALIASES:
        return OBJECTIVE_ALIASES[s]
    return s.lower().replace(" ", "_").replace("-", "_")


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "objective_key" in df.columns and "objective" not in df.columns:
        df = df.rename(columns={"objective_key": "objective"})

    if "learning_type" in df.columns and "objective" not in df.columns:
        df = df.rename(columns={"learning_type": "objective"})

    if "localization_condition" in df.columns and "condition" not in df.columns:
        df = df.rename(columns={"localization_condition": "condition"})

    if "generalization" in df.columns and "transfer" not in df.columns:
        df = df.rename(columns={"generalization": "transfer"})

    if "acquisition" not in df.columns:
        if {"id_eval", "paraphrase_eval"}.issubset(df.columns):
            df["acquisition"] = (
                pd.to_numeric(df["id_eval"], errors="coerce")
                + pd.to_numeric(df["paraphrase_eval"], errors="coerce")
            ) / 2
        else:
            raise ValueError(
                "Input must contain acquisition or both id_eval and paraphrase_eval."
            )

    required = {"objective", "condition", "seed", "acquisition", "transfer", "boundedness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns)}"
        )

    df["objective"] = df["objective"].map(normalize_objective_name)
    df["condition"] = df["condition"].astype(str).str.lower()
    df = df[df["condition"].isin(CONDITIONS)].copy()

    for metric in PRIMARY_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["balanced_mean"] = df[PRIMARY_METRICS].mean(axis=1)
    df["balanced_min"] = df[PRIMARY_METRICS].min(axis=1)

    df["seed"] = df["seed"].astype(str)

    if "control" not in df.columns:
        df["control"] = "main"

    return df


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    observed = float(values.mean())

    if len(values) == 1:
        return observed, observed, observed, np.nan

    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    # Bootstrap sign test. With only 3 seeds, this is descriptive.
    p_two = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p_two = min(float(p_two), 1.0)

    return observed, float(ci_low), float(ci_high), p_two


def compute_by_seed_penalties(
    df: pd.DataFrame,
    contrasts: List[Dict[str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for control, df_control in df.groupby("control"):
        for spec in contrasts:
            objective = spec["objective"]
            preferred = spec["preferred_condition"]
            mislocated = spec["mislocated_condition"]

            sub = df_control[df_control["objective"] == objective].copy()
            if sub.empty:
                continue

            # One value per seed x condition.
            agg = (
                sub.groupby(["seed", "condition"], as_index=False)[ALL_METRICS]
                .mean()
            )

            preferred_df = agg[agg["condition"] == preferred].set_index("seed")
            mislocated_df = agg[agg["condition"] == mislocated].set_index("seed")
            common_seeds = sorted(set(preferred_df.index) & set(mislocated_df.index))

            for seed in common_seeds:
                base = {
                    "control": control,
                    "objective": objective,
                    "contrast_id": spec["contrast_id"],
                    "component": spec["component"],
                    "preferred_condition": preferred,
                    "mislocated_condition": mislocated,
                    "seed": seed,
                    "interpretation": spec["interpretation"],
                }

                for metric in ALL_METRICS:
                    pref_value = float(preferred_df.loc[seed, metric])
                    mis_value = float(mislocated_df.loc[seed, metric])
                    base[f"{metric}_preferred"] = pref_value
                    base[f"{metric}_mislocated"] = mis_value
                    base[f"{metric}_penalty"] = pref_value - mis_value
                    if pref_value != 0:
                        base[f"{metric}_retention"] = mis_value / pref_value
                    else:
                        base[f"{metric}_retention"] = np.nan

                rows.append(base)

    return pd.DataFrame(rows)


def summarize_penalties(
    by_seed: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_cols = [
        "control",
        "objective",
        "contrast_id",
        "component",
        "preferred_condition",
        "mislocated_condition",
        "interpretation",
    ]

    for keys, group in by_seed.groupby(group_cols):
        key_dict = dict(zip(group_cols, keys))

        for metric in ALL_METRICS:
            penalty_col = f"{metric}_penalty"
            pref_col = f"{metric}_preferred"
            mis_col = f"{metric}_mislocated"
            retention_col = f"{metric}_retention"

            penalties = group[penalty_col].dropna().to_numpy(float)
            mean_delta, ci_low, ci_high, p_boot = bootstrap_mean_ci(
                penalties, n_boot=n_boot, rng=rng
            )

            rows.append({
                **key_dict,
                "metric": metric,
                "preferred_mean": group[pref_col].mean(),
                "mislocated_mean": group[mis_col].mean(),
                "mislocation_penalty": mean_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_boot_two_sided": p_boot,
                "sign_rate_positive": float(np.mean(penalties > 0)) if len(penalties) else np.nan,
                "retention_mean": group[retention_col].mean(),
                "n_paired_seeds": len(penalties),
            })

    return pd.DataFrame(rows)


def make_wide_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_cols = [
        "control",
        "objective",
        "contrast_id",
        "component",
        "preferred_condition",
        "mislocated_condition",
        "interpretation",
    ]

    for keys, group in summary.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for metric in ALL_METRICS:
            m = group[group["metric"] == metric]
            if m.empty:
                continue
            r = m.iloc[0]
            row[f"{metric}_preferred_mean"] = r["preferred_mean"]
            row[f"{metric}_mislocated_mean"] = r["mislocated_mean"]
            row[f"{metric}_penalty"] = r["mislocation_penalty"]
            row[f"{metric}_ci_low"] = r["ci_low"]
            row[f"{metric}_ci_high"] = r["ci_high"]
            row[f"{metric}_retention_mean"] = r["retention_mean"]

        rows.append(row)

    return pd.DataFrame(rows)


def fmt_ci(row: pd.Series, metric: str) -> str:
    delta = row.get(f"{metric}_penalty", np.nan)
    lo = row.get(f"{metric}_ci_low", np.nan)
    hi = row.get(f"{metric}_ci_high", np.nan)

    if pd.isna(delta):
        return "--"

    return f"{delta:.3f} [{lo:.3f}, {hi:.3f}]"


def latex_escape(s: Any) -> str:
    return str(s).replace("_", r"\_")


def write_latex_table(wide: pd.DataFrame, path: Path) -> None:
    metrics_for_table = ["acquisition", "transfer", "boundedness", "balanced_mean"]

    lines = []
    lines.append(r"\begin{table*}[t]" + "\n")
    lines.append(r"\centering" + "\n")
    lines.append(r"\scriptsize" + "\n")
    lines.append(r"\setlength{\tabcolsep}{3pt}" + "\n")
    lines.append(r"\begin{tabular}{lllcccc}" + "\n")
    lines.append(r"\toprule" + "\n")
    lines.append(
        r"Objective & Preferred & Mislocated & Acquisition $\Delta$ & Transfer $\Delta$ & Boundedness $\Delta$ & Balanced $\Delta$ \\"
        + "\n"
    )
    lines.append(r"\midrule" + "\n")

    for _, row in wide.iterrows():
        objective = latex_escape(row["objective"])
        preferred = latex_escape(row["preferred_condition"])
        mislocated = latex_escape(row["mislocated_condition"])

        vals = [fmt_ci(row, m) for m in metrics_for_table]

        lines.append(
            f"{objective} & {preferred} & {mislocated} & "
            + " & ".join(vals)
            + r" \\"
            + "\n"
        )

    lines.append(r"\bottomrule" + "\n")
    lines.append(r"\end{tabular}" + "\n")
    lines.append(
        r"\caption{Mislocation penalties computed from localized adaptation results. "
        r"Each penalty is preferred condition minus mislocated condition. "
        r"Positive values indicate that respecting the objective's adaptation geometry improves performance. "
        r"Intervals are bootstrap confidence intervals over paired seed-level differences.}"
        + "\n"
    )
    lines.append(r"\label{tab:mislocation-penalties}" + "\n")
    lines.append(r"\end{table*}" + "\n")

    path.write_text("".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame, wide: pd.DataFrame, path: Path) -> None:
    lines: List[str] = []
    lines.append("# Mislocation Penalty Analysis\n\n")
    lines.append(
        "Mislocation penalty is defined as preferred condition minus mislocated condition. "
        "Positive values indicate that respecting the adaptation geometry improves performance.\n\n"
    )

    lines.append("## Strongest positive penalties\n\n")
    top = summary.dropna(subset=["mislocation_penalty"]).copy()
    top = top.sort_values("mislocation_penalty", ascending=False).head(20)

    for _, row in top.iterrows():
        lines.append(
            f"- **{row['objective']} / {row['metric']} / {row['contrast_id']}**: "
            f"Δ={row['mislocation_penalty']:.3f}, "
            f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
            f"preferred={row['preferred_condition']}, "
            f"mislocated={row['mislocated_condition']}, "
            f"n={int(row['n_paired_seeds'])}\n"
        )

    lines.append("\n## Contrast-level summary\n\n")
    for _, row in wide.iterrows():
        lines.append(f"### {row['contrast_id']}\n\n")
        lines.append(f"- Objective: `{row['objective']}`\n")
        lines.append(f"- Component: {row['component']}\n")
        lines.append(f"- Preferred: `{row['preferred_condition']}`\n")
        lines.append(f"- Mislocated: `{row['mislocated_condition']}`\n")
        lines.append(f"- Interpretation: {row['interpretation']}\n")
        lines.append(f"- Acquisition penalty: {fmt_ci(row, 'acquisition')}\n")
        lines.append(f"- Transfer penalty: {fmt_ci(row, 'transfer')}\n")
        lines.append(f"- Boundedness penalty: {fmt_ci(row, 'boundedness')}\n")
        lines.append(f"- Balanced mean penalty: {fmt_ci(row, 'balanced_mean')}\n\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mislocation_analysis"))
    parser.add_argument(
        "--control",
        default=None,
        help="Optional filter for parameter-matched files with a control column.",
    )
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = normalize_input(df)

    if args.control is not None:
        df = df[df["control"] == args.control].copy()

    by_seed = compute_by_seed_penalties(df, MISLOCATION_CONTRASTS)
    if by_seed.empty:
        raise RuntimeError(
            "No mislocation contrasts could be computed. "
            "Check objective names, condition names, and seed coverage."
        )

    summary_long = summarize_penalties(by_seed, n_boot=args.n_boot, rng=rng)
    summary_wide = make_wide_summary(summary_long)

    by_seed_path = args.output_dir / "mislocation_penalties_by_seed.csv"
    long_path = args.output_dir / "mislocation_penalty_summary_long.csv"
    wide_path = args.output_dir / "mislocation_penalty_summary_wide.csv"
    tex_path = args.output_dir / "mislocation_penalty_table.tex"
    report_path = args.output_dir / "mislocation_report.md"

    by_seed.to_csv(by_seed_path, index=False)
    summary_long.to_csv(long_path, index=False)
    summary_wide.to_csv(wide_path, index=False)
    write_latex_table(summary_wide, tex_path)
    write_report(summary_long, summary_wide, report_path)

    print(f"Wrote {by_seed_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
