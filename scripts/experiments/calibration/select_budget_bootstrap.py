#!/usr/bin/env python3
"""
Select a calibrated budget using a hierarchical bootstrap over seeds and latent specs.

Updated version:
  - Separates concept transfer from yes/no policy transfer.
  - Default selection uses concept transfer, not ALL_MODES generalization.
  - Reports both:
      concept_transfer = generalization_concept strict accuracy
      policy_transfer  = generalization_yes_no strict accuracy
      generalization_all = all generalization rows combined

Why:
  Lexical binding showed a clean dissociation:
    concept transfer improves with budget, while yes/no generalization can decline.
  So the calibrated budget should not fail simply because ALL_MODES generalization
  mixes concept transfer with response-policy transfer.

Inputs:
  Raw adapter JSONL files, e.g.
    outputs/multiseed_lexical/adapter_calib_lexical_binding_specs25_budget8_seed11.jsonl

Example:

  python scripts/experiments/calibration/select_budget_bootstrap.py \
    --inputs outputs/multiseed_lexical/adapter_*.jsonl \
    --output_csv outputs/multiseed_lexical/budget_selection_bootstrap.csv \
    --decision_json outputs/multiseed_lexical/budget_selection_decision.json \
    --figure_dir outputs/multiseed_lexical/figures_bootstrap \
    --acquisition_threshold 0.85 \
    --transfer_threshold 0.80 \
    --transfer_metric concept_transfer

Metric definitions:
  acquisition       = mean(ID eval strict, paraphrase strict)
  concept_transfer  = strict accuracy on generalization_concept rows
  policy_transfer   = strict accuracy on generalization_yes_no rows
  generalization_all = strict accuracy on all generalization rows
  boundedness       = strict accuracy on negative_control rows

Selection rule:
  choose the smallest budget whose lower CI clears:
    acquisition >= acquisition_threshold
    chosen transfer metric >= transfer_threshold

Optional:
  --boundedness_threshold X
  --policy_transfer_threshold X
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--inputs", nargs="+", required=True, help="Raw adapter JSONL files.")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--decision_json", required=True)
    p.add_argument("--figure_dir", default=None)

    p.add_argument("--acquisition_threshold", type=float, default=0.85)
    p.add_argument("--transfer_threshold", type=float, default=0.80)
    p.add_argument(
        "--transfer_metric",
        choices=["concept_transfer", "policy_transfer", "generalization_all"],
        default="concept_transfer",
        help="Which transfer metric is used for budget selection.",
    )
    p.add_argument("--boundedness_threshold", type=float, default=None)
    p.add_argument("--policy_transfer_threshold", type=float, default=None)

    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--random_seed", type=int, default=1234)

    p.add_argument(
        "--unit",
        choices=["latent_spec", "prompt"],
        default="latent_spec",
        help="Resampling unit within each seed. latent_spec is recommended.",
    )
    return p.parse_args()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def infer_from_filename(path: str | Path) -> Dict[str, Any]:
    name = Path(path).name

    m_budget = re.search(r"budget(\d+)", name)
    m_seed = re.search(r"seed(\d+)", name)
    m_specs = re.search(r"specs(\d+)", name)
    m_lt = re.search(r"calib_(.*?)_specs\d+_budget\d+", name)

    return {
        "budget": int(m_budget.group(1)) if m_budget else None,
        "seed": int(m_seed.group(1)) if m_seed else None,
        "n_specs": int(m_specs.group(1)) if m_specs else None,
        "learning_type": m_lt.group(1) if m_lt else None,
    }


def get_nested(row: Dict[str, Any], *keys: str):
    cur = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def row_to_record(row: Dict[str, Any], meta: Dict[str, Any], path: str | Path) -> Dict[str, Any]:
    spec_id = (
        row.get("spec_id")
        or row.get("latent_spec_id")
        or get_nested(row, "metadata", "spec_id")
        or get_nested(row, "metadata", "latent_spec_id")
        or get_nested(row, "latent_spec", "spec_id")
        or get_nested(row, "latent_spec", "id")
        or row.get("example_id")
    )

    example_mode = (
        row.get("example_mode")
        or get_nested(row, "metadata", "example_mode")
        or "unknown"
    )

    return {
        "file": Path(path).name,
        "learning_type": row.get("learning_type") or meta["learning_type"] or "unknown",
        "budget": row.get("train_budget_per_spec") or meta["budget"],
        "seed": meta["seed"],
        "split": row.get("split", "unknown"),
        "example_mode": example_mode,
        "spec_id": spec_id,
        "example_id": row.get("example_id"),
        "strict_score": int(bool(row.get("strict_score", row.get("passed", False)))),
        "loose_score": int(bool(row.get("loose_score", row.get("passed", False)))),
        "concept_score": int(bool(row.get("concept_score", row.get("target_mentioned", False)))),
    }


def load_records(paths: List[str | Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        meta = infer_from_filename(path)
        if meta["budget"] is None:
            raise ValueError(f"Could not infer budget from filename: {path}")
        if meta["seed"] is None:
            raise ValueError(
                f"Could not infer seed from filename: {path}\n"
                "Expected filenames containing ..._seed11.jsonl etc."
            )
        for row in read_jsonl(path):
            records.append(row_to_record(row, meta, path))

    df = pd.DataFrame(records)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    return df


def safe_mean(vals: pd.Series) -> float:
    if vals.empty:
        return np.nan
    return float(vals.mean())


def make_unit_metric_table(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """
    Produce one row per learning_type/budget/seed/unit_id with metric columns.
    """
    unit_col = "spec_id" if unit == "latent_spec" else "example_id"

    rows = []

    group_cols = ["learning_type", "budget", "seed", unit_col]
    for (learning_type, budget, seed, unit_id), group in df.groupby(group_cols):
        id_eval = safe_mean(group.loc[group["split"] == "id_eval", "strict_score"])
        paraphrase = safe_mean(group.loc[group["split"] == "paraphrase_eval", "strict_score"])
        generalization_all = safe_mean(group.loc[group["split"] == "generalization", "strict_score"])
        boundedness = safe_mean(group.loc[group["split"] == "negative_control", "strict_score"])

        concept_transfer = safe_mean(
            group.loc[
                (group["split"] == "generalization")
                & (group["example_mode"] == "generalization_concept"),
                "strict_score",
            ]
        )
        policy_transfer = safe_mean(
            group.loc[
                (group["split"] == "generalization")
                & (group["example_mode"] == "generalization_yes_no"),
                "strict_score",
            ]
        )

        acquisition = np.nanmean([id_eval, paraphrase])

        rows.append({
            "learning_type": learning_type,
            "budget": int(budget),
            "seed": int(seed),
            "unit_id": unit_id,
            "id_eval": id_eval,
            "paraphrase_eval": paraphrase,
            "acquisition": acquisition,
            "generalization_all": generalization_all,
            "concept_transfer": concept_transfer,
            "policy_transfer": policy_transfer,
            "boundedness": boundedness,
            "negative_control": boundedness,
        })

    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    budget_df: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    seeds = sorted(budget_df["seed"].dropna().unique())

    if len(seeds) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    seed_means = []
    for seed in seeds:
        vals = budget_df.loc[budget_df["seed"] == seed, metric].dropna().to_numpy()
        if len(vals):
            seed_means.append(float(vals.mean()))
    observed = float(np.mean(seed_means)) if seed_means else np.nan

    boot = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_seed_means = []

        for seed in sampled_seeds:
            vals = budget_df.loc[budget_df["seed"] == seed, metric].dropna().to_numpy()
            if len(vals) == 0:
                continue

            sampled_vals = rng.choice(vals, size=len(vals), replace=True)
            sampled_seed_means.append(float(sampled_vals.mean()))

        boot[i] = np.mean(sampled_seed_means) if sampled_seed_means else np.nan

    alpha = 1.0 - ci
    low = float(np.nanpercentile(boot, 100 * alpha / 2))
    high = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))

    return {"mean": observed, "ci_low": low, "ci_high": high}


def summarize(unit_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_seed)

    metrics = [
        "acquisition",
        "concept_transfer",
        "policy_transfer",
        "generalization_all",
        "boundedness",
        "id_eval",
        "paraphrase_eval",
        "negative_control",
    ]

    rows = []
    for (learning_type, budget), group in unit_df.groupby(["learning_type", "budget"]):
        n_seeds = group["seed"].nunique()
        n_units = group["unit_id"].nunique()

        for metric in metrics:
            stats = hierarchical_bootstrap(
                group,
                metric=metric,
                n_bootstrap=args.n_bootstrap,
                ci=args.ci,
                rng=rng,
            )
            rows.append({
                "learning_type": learning_type,
                "budget": int(budget),
                "metric": metric,
                "mean": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "n_seeds": n_seeds,
                "n_units": n_units,
                "ci": args.ci,
                "n_bootstrap": args.n_bootstrap,
                "unit": args.unit,
            })

    return pd.DataFrame(rows).sort_values(["learning_type", "budget", "metric"])


def make_decision(summary_df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    decisions = {}

    for learning_type in sorted(summary_df["learning_type"].unique()):
        sub = summary_df[summary_df["learning_type"] == learning_type]
        budgets = sorted(sub["budget"].unique())

        budget_rows = []
        for budget in budgets:
            bdf = sub[sub["budget"] == budget].set_index("metric")

            acquisition_low = float(bdf.loc["acquisition", "ci_low"])
            transfer_low = float(bdf.loc[args.transfer_metric, "ci_low"])
            boundedness_low = float(bdf.loc["boundedness", "ci_low"])
            policy_low = float(bdf.loc["policy_transfer", "ci_low"])

            passes = (
                acquisition_low >= args.acquisition_threshold
                and transfer_low >= args.transfer_threshold
            )

            if args.boundedness_threshold is not None:
                passes = passes and boundedness_low >= args.boundedness_threshold

            if args.policy_transfer_threshold is not None:
                passes = passes and policy_low >= args.policy_transfer_threshold

            budget_rows.append({
                "budget": int(budget),
                "acquisition_ci_low": acquisition_low,
                f"{args.transfer_metric}_ci_low": transfer_low,
                "boundedness_ci_low": boundedness_low,
                "policy_transfer_ci_low": policy_low,
                "passes": bool(passes),
            })

        passing = [r["budget"] for r in budget_rows if r["passes"]]
        selected = min(passing) if passing else None

        decisions[learning_type] = {
            "selected_budget": selected,
            "candidate_budgets": [int(b) for b in budgets],
            "selection_rule": {
                "choose": "smallest budget passing all required thresholds",
                "acquisition_lower_ci_threshold": args.acquisition_threshold,
                "transfer_metric": args.transfer_metric,
                "transfer_lower_ci_threshold": args.transfer_threshold,
                "boundedness_lower_ci_threshold": args.boundedness_threshold,
                "policy_transfer_lower_ci_threshold": args.policy_transfer_threshold,
                "ci": args.ci,
                "unit": args.unit,
            },
            "budget_checks": budget_rows,
        }

    return decisions


def save_figures(summary_df: pd.DataFrame, figure_dir: str | Path, args: argparse.Namespace) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics = [
        "acquisition",
        "concept_transfer",
        "policy_transfer",
        "boundedness",
    ]
    metric_labels = {
        "acquisition": "Acquisition",
        "concept_transfer": "Concept transfer",
        "policy_transfer": "Policy transfer",
        "boundedness": "Boundedness",
    }

    for learning_type in sorted(summary_df["learning_type"].unique()):
        sub = summary_df[
            (summary_df["learning_type"] == learning_type)
            & (summary_df["metric"].isin(plot_metrics))
        ].copy()

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.0, 5.4))

        for metric in plot_metrics:
            mdf = sub[sub["metric"] == metric].sort_values("budget")
            x = mdf["budget"].to_numpy()
            y = mdf["mean"].to_numpy()
            low = mdf["ci_low"].to_numpy()
            high = mdf["ci_high"].to_numpy()
            yerr = np.vstack([y - low, high - y])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=metric_labels[metric],
            )

        ax.axhline(args.acquisition_threshold, linestyle="--", linewidth=1, label="Acquisition threshold")
        ax.axhline(args.transfer_threshold, linestyle=":", linewidth=1, label="Transfer threshold")

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training budget per latent specification")
        ax.set_ylabel("Strict accuracy")
        ax.set_title(f"Bootstrap budget selection: {learning_type}")
        ax.legend(frameon=True)
        fig.tight_layout()

        png = figure_dir / f"{learning_type}_bootstrap_budget_selection.png"
        pdf = figure_dir / f"{learning_type}_bootstrap_budget_selection.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote {png}")
        print(f"Wrote {pdf}")


def main() -> None:
    args = parse_args()

    df = load_records(args.inputs)
    unit_df = make_unit_metric_table(df, unit=args.unit)

    summary_df = summarize(unit_df, args)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    decisions = make_decision(summary_df, args)
    decision_path = Path(args.decision_json)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print(f"Wrote {decision_path}")

    print()
    print("Budget decision:")
    for lt, decision in decisions.items():
        print(f"  {lt}: selected_budget={decision['selected_budget']}")

    if args.figure_dir:
        save_figures(summary_df, args.figure_dir, args)


if __name__ == "__main__":
    main()
