#!/usr/bin/env python3
"""
Analyze localization statistics for adaptation geometry experiments.

Expected input: a seed-level CSV with columns like:
  objective, condition, seed, acquisition, transfer, boundedness

Also accepts:
  localization_condition instead of condition
  generalization instead of transfer
  id_eval + paraphrase_eval instead of acquisition

Outputs:
  localization_descriptive_stats.csv
  localization_planned_contrasts.csv
  adaptation_geometry_metrics.csv
  adaptation_geometry_distances.csv
  geometry_omnibus_test.json
  localization_statistics_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CONDITIONS = ["full", "early", "middle", "late"]
LOCALIZED = ["early", "middle", "late"]
METRICS = ["acquisition", "transfer", "boundedness"]


PLANNED_CONTRASTS = [
    # Lexical: early-localizable acquisition, but transfer/deployment tradeoff.
    ("lexical_binding", "acquisition", "early", "late", "early_vs_late_acquisition"),
    ("lexical_binding", "acquisition", "early", "middle", "early_vs_middle_acquisition"),
    ("lexical_binding", "boundedness", "early", "full", "early_vs_full_boundedness"),
    ("lexical_binding", "transfer", "full", "early", "full_vs_early_transfer"),

    # Factual: late/full better than early for relational association.
    ("factual_association", "acquisition", "late", "early", "late_vs_early_acquisition"),
    ("factual_association", "transfer", "late", "early", "late_vs_early_transfer"),
    ("factual_association", "transfer", "full", "late", "full_vs_late_transfer"),
    ("factual_association", "boundedness", "late", "full", "late_vs_full_boundedness"),

    # Behavioral: distributed; middle often better bounded/generalized than late.
    ("behavioral_policy", "boundedness", "middle", "late", "middle_vs_late_boundedness"),
    ("behavioral_policy", "transfer", "middle", "late", "middle_vs_late_transfer"),
    ("behavioral_policy", "acquisition", "late", "middle", "late_vs_middle_acquisition"),
    ("behavioral_policy", "boundedness", "full", "middle", "full_vs_middle_boundedness"),

    # Causal: middle strongest localized; full strongest transfer.
    ("causal_mapping", "transfer", "middle", "early", "middle_vs_early_transfer"),
    ("causal_mapping", "transfer", "middle", "late", "middle_vs_late_transfer"),
    ("causal_mapping", "transfer", "full", "middle", "full_vs_middle_transfer"),
    ("causal_mapping", "boundedness", "middle", "early", "middle_vs_early_boundedness"),

    # Procedural: middle balanced; late bounded but conservative.
    ("procedural_reasoning", "transfer", "middle", "early", "middle_vs_early_transfer"),
    ("procedural_reasoning", "transfer", "full", "middle", "full_vs_middle_transfer"),
    ("procedural_reasoning", "boundedness", "late", "middle", "late_vs_middle_boundedness"),
    ("procedural_reasoning", "acquisition", "middle", "late", "middle_vs_late_acquisition"),
]


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
                df["id_eval"].astype(float) + df["paraphrase_eval"].astype(float)
            ) / 2
        else:
            raise ValueError("Input must contain acquisition or both id_eval and paraphrase_eval.")

    required = {"objective", "condition", "seed", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns)}"
        )

    df["condition"] = df["condition"].astype(str).str.lower()
    df = df[df["condition"].isin(CONDITIONS)].copy()

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["seed"] = df["seed"].astype(str)
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def descriptive_stats(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for (objective, condition), group in df.groupby(["objective", "condition"]):
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            ci_low, ci_high = bootstrap_ci(values, n_boot, rng)

            rows.append({
                "objective": objective,
                "condition": condition,
                "metric": metric,
                "mean": np.mean(values) if len(values) else np.nan,
                "sd": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_seeds": len(values),
            })

    return pd.DataFrame(rows)


def paired_contrast(
    df: pd.DataFrame,
    objective: str,
    metric: str,
    condition_a: str,
    condition_b: str,
    label: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict:
    sub = df[df["objective"] == objective]
    pivot = sub.pivot_table(index="seed", columns="condition", values=metric, aggfunc="mean")

    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return {
            "objective": objective,
            "metric": metric,
            "contrast": label,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "delta_a_minus_b": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_boot_two_sided": np.nan,
            "n_paired_seeds": 0,
            "note": "missing condition",
        }

    paired = pivot[[condition_a, condition_b]].dropna()
    if len(paired) == 0:
        return {
            "objective": objective,
            "metric": metric,
            "contrast": label,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "delta_a_minus_b": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_boot_two_sided": np.nan,
            "n_paired_seeds": 0,
            "note": "no paired seeds",
        }

    diffs = (paired[condition_a] - paired[condition_b]).to_numpy(float)
    observed = float(np.mean(diffs))

    if len(diffs) == 1:
        boot = np.array([observed])
    else:
        boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_two = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p_two = min(float(p_two), 1.0)

    return {
        "objective": objective,
        "metric": metric,
        "contrast": label,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "delta_a_minus_b": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_boot_two_sided": p_two,
        "n_paired_seeds": len(diffs),
        "note": "",
    }


def planned_contrasts(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = [
        paired_contrast(df, obj, metric, a, b, label, n_boot, rng)
        for obj, metric, a, b, label in PLANNED_CONTRASTS
    ]
    return pd.DataFrame(rows)


def geometry_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lightweight adaptation geometry metrics.

    selectivity_all:
      max condition performance - min condition performance

    selectivity_localized:
      max localized performance - min localized performance

    localized_advantage:
      best localized condition - full

    best_localized_condition:
      condition among early/middle/late with highest score

    balanced_mean:
      mean(acquisition, transfer, boundedness) per objective/condition

    balanced_min:
      min(acquisition, transfer, boundedness) per objective/condition
    """
    means = (
        df.groupby(["objective", "condition"])[METRICS]
        .mean()
        .reset_index()
    )

    rows = []

    for objective, obj_df in means.groupby("objective"):
        for metric in METRICS:
            cond_scores = {
                row["condition"]: row[metric]
                for _, row in obj_df.iterrows()
                if row["condition"] in CONDITIONS
            }

            if "full" not in cond_scores:
                continue

            localized_scores = {c: cond_scores[c] for c in LOCALIZED if c in cond_scores}
            if not localized_scores:
                continue

            best_loc = max(localized_scores, key=localized_scores.get)
            best_loc_value = localized_scores[best_loc]

            rows.append({
                "objective": objective,
                "metric": metric,
                "selectivity_all": max(cond_scores.values()) - min(cond_scores.values()),
                "selectivity_localized": max(localized_scores.values()) - min(localized_scores.values()),
                "localized_advantage": best_loc_value - cond_scores["full"],
                "best_localized_condition": best_loc,
                "best_localized_value": best_loc_value,
                "full_value": cond_scores["full"],
            })

    # Add balanced scores per condition.
    balanced_rows = []
    for _, row in means.iterrows():
        vals = [row[m] for m in METRICS]
        balanced_rows.append({
            "objective": row["objective"],
            "metric": "balanced",
            "condition": row["condition"],
            "balanced_mean": float(np.mean(vals)),
            "balanced_min": float(np.min(vals)),
        })

    geom = pd.DataFrame(rows)
    balanced = pd.DataFrame(balanced_rows)
    return geom, balanced


def make_seed_profiles(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build one adaptation-geometry profile per objective x seed.

    Profile vector:
      [early-full, middle-full, late-full] x [acquisition, transfer, boundedness]

    This focuses on geometry relative to full-stack adaptation, rather than raw task difficulty.
    """
    profiles = []
    labels = []

    grouped = df.groupby(["objective", "seed"])
    for (objective, seed), group in grouped:
        pivot = group.pivot_table(index="condition", values=METRICS, aggfunc="mean")

        if "full" not in pivot.index:
            continue
        if not all(c in pivot.index for c in LOCALIZED):
            continue

        vec = []
        for metric in METRICS:
            full = pivot.loc["full", metric]
            for cond in LOCALIZED:
                vec.append(pivot.loc[cond, metric] - full)

        if np.any(pd.isna(vec)):
            continue

        profiles.append(vec)
        labels.append(objective)

    return np.asarray(profiles, dtype=float), labels


def heterogeneity_statistic(X: np.ndarray, labels: List[str]) -> float:
    """
    Between-objective variance of geometry profiles.

    Larger values mean objectives have more distinct adaptation geometries.
    """
    labels_arr = np.asarray(labels)
    grand = X.mean(axis=0)

    stat = 0.0
    objectives = sorted(set(labels))
    for obj in objectives:
        X_obj = X[labels_arr == obj]
        if len(X_obj) == 0:
            continue
        mu = X_obj.mean(axis=0)
        stat += float(np.sum((mu - grand) ** 2))

    return stat / max(len(objectives), 1)


def geometry_omnibus_test(
    df: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict:
    X, labels = make_seed_profiles(df)

    if len(X) == 0:
        return {
            "profile_definition": "[early-full, middle-full, late-full] x acquisition/transfer/boundedness",
            "observed_geometry_heterogeneity": None,
            "permutation_p": None,
            "n_profiles": 0,
            "note": "No complete seed-level profiles found.",
        }

    observed = heterogeneity_statistic(X, labels)

    perm_stats = []
    labels_arr = np.asarray(labels)
    for _ in range(n_perm):
        permuted = rng.permutation(labels_arr)
        perm_stats.append(heterogeneity_statistic(X, list(permuted)))

    perm_stats = np.asarray(perm_stats)
    p = float((np.sum(perm_stats >= observed) + 1) / (n_perm + 1))

    return {
        "profile_definition": "[early-full, middle-full, late-full] x acquisition/transfer/boundedness",
        "observed_geometry_heterogeneity": float(observed),
        "permutation_p": p,
        "n_profiles": int(len(X)),
        "n_permutations": int(n_perm),
        "note": (
            "Permutation test treats objective labels as exchangeable across seed-level "
            "geometry profiles. Interpret as an omnibus diagnostic, not as a replacement "
            "for planned contrasts."
        ),
    }


def objective_profile_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise distances between objective-level adaptation geometry profiles.

    Uses mean profile per objective:
      [early-full, middle-full, late-full] x [acquisition, transfer, boundedness]
    """
    X, labels = make_seed_profiles(df)
    if len(X) == 0:
        return pd.DataFrame()

    profile_df = pd.DataFrame(X)
    profile_df["objective"] = labels
    means = profile_df.groupby("objective").mean()

    rows = []
    objectives = list(means.index)
    for i, obj_i in enumerate(objectives):
        for obj_j in objectives[i + 1:]:
            vi = means.loc[obj_i].to_numpy(float)
            vj = means.loc[obj_j].to_numpy(float)
            rows.append({
                "objective_i": obj_i,
                "objective_j": obj_j,
                "geometry_distance": float(np.linalg.norm(vi - vj)),
            })

    return pd.DataFrame(rows).sort_values("geometry_distance", ascending=False)


def write_report(
    path: Path,
    desc: pd.DataFrame,
    contrasts: pd.DataFrame,
    geom: pd.DataFrame,
    omnibus: Dict,
) -> None:
    lines = []
    lines.append("# Localization Statistics Report\n")

    lines.append("## Omnibus adaptation-geometry test\n")
    lines.append(
        f"- Observed geometry heterogeneity: "
        f"{omnibus.get('observed_geometry_heterogeneity')}\n"
    )
    lines.append(f"- Permutation p-value: {omnibus.get('permutation_p')}\n")
    lines.append(f"- Profiles: {omnibus.get('n_profiles')}\n")
    lines.append(f"- Note: {omnibus.get('note')}\n\n")

    lines.append("## Strongest planned contrasts by absolute effect size\n")
    c = contrasts.dropna(subset=["delta_a_minus_b"]).copy()
    if len(c):
        c["abs_delta"] = c["delta_a_minus_b"].abs()
        top = c.sort_values("abs_delta", ascending=False).head(12)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['objective']} / {row['metric']} / {row['contrast']}: "
                f"Δ={row['delta_a_minus_b']:.3f}, "
                f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                f"p_boot={row['p_boot_two_sided']:.3f}, "
                f"n={int(row['n_paired_seeds'])}\n"
            )
    lines.append("\n")

    lines.append("## Geometry metrics\n")
    if len(geom):
        for _, row in geom.iterrows():
            lines.append(
                f"- {row['objective']} / {row['metric']}: "
                f"selectivity_all={row['selectivity_all']:.3f}, "
                f"selectivity_localized={row['selectivity_localized']:.3f}, "
                f"localized_advantage={row['localized_advantage']:.3f}, "
                f"best_localized={row['best_localized_condition']}\n"
            )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/localization_statistics"))
    parser.add_argument("--control", default=None, help="Optional filter if input has a control column.")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = normalize_input(df)

    if args.control is not None and "control" in df.columns:
        df = df[df["control"] == args.control].copy()

    desc = descriptive_stats(df, args.n_boot, rng)
    contrasts = planned_contrasts(df, args.n_boot, rng)
    geom, balanced = geometry_metrics(df)
    distances = objective_profile_distances(df)
    omnibus = geometry_omnibus_test(df, args.n_perm, rng)

    desc.to_csv(args.output_dir / "localization_descriptive_stats.csv", index=False)
    contrasts.to_csv(args.output_dir / "localization_planned_contrasts.csv", index=False)
    geom.to_csv(args.output_dir / "adaptation_geometry_metrics.csv", index=False)
    balanced.to_csv(args.output_dir / "adaptation_balanced_scores.csv", index=False)
    distances.to_csv(args.output_dir / "adaptation_geometry_distances.csv", index=False)

    with open(args.output_dir / "geometry_omnibus_test.json", "w", encoding="utf-8") as f:
        json.dump(omnibus, f, indent=2)

    write_report(
        args.output_dir / "localization_statistics_report.md",
        desc,
        contrasts,
        geom,
        omnibus,
    )

    print(f"Wrote outputs to {args.output_dir}")
    print("\nKey files:")
    print(args.output_dir / "localization_descriptive_stats.csv")
    print(args.output_dir / "localization_planned_contrasts.csv")
    print(args.output_dir / "adaptation_geometry_metrics.csv")
    print(args.output_dir / "geometry_omnibus_test.json")
    print(args.output_dir / "localization_statistics_report.md")


if __name__ == "__main__":
    main()
