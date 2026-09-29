#!/usr/bin/env python3
"""
Compute a relation-type breakdown for factual-association calibration runs.

This script is meant to answer questions like:

  - Is capital_city easier than invented_device?
  - Is affiliated_institute dragging down ID retrieval?
  - Do different factual relation types fail differently on negative controls?
  - Does the budget8 -> budget12 jump come from one relation subtype?

It groups factual-association raw JSONL outputs by:

  condition      baseline / adapter
  budget
  split          id_eval / paraphrase_eval / generalization / negative_control
  example_mode   retrieval / positive_polarity / negative_polarity / generalization_concept
  scoring_type
  relation_type  capital_city / invented_device / affiliated_institute

It can also write simple appendix-style figures.

Example:

  python scripts/reproduce/factual_relation_breakdown.py \
    --inputs outputs/mode_aware_budget_sweep_factual/*.jsonl \
    --examples_path data/prompt_examples.jsonl \
    --output_csv outputs/mode_aware_budget_sweep_factual/factual_relation_breakdown.csv \
    --figure_dir outputs/mode_aware_budget_sweep_factual/figures_relation_breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


RELATION_ORDER = ["capital_city", "invented_device", "affiliated_institute"]
SPLIT_ORDER = ["id_eval", "paraphrase_eval", "generalization", "negative_control"]
MODE_ORDER = ["retrieval", "positive_polarity", "negative_polarity", "generalization_concept"]

SPLIT_LABELS = {
    "id_eval": "ID eval",
    "paraphrase_eval": "Paraphrase",
    "generalization": "Generalization",
    "negative_control": "Negative control",
}

RELATION_LABELS = {
    "capital_city": "Capital city",
    "invented_device": "Invented device",
    "affiliated_institute": "Affiliated institute",
}

MODE_LABELS = {
    "retrieval": "Retrieval",
    "positive_polarity": "Positive polarity",
    "negative_polarity": "Negative polarity",
    "generalization_concept": "Generalization",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Raw baseline/adapter JSONL files.")
    p.add_argument(
        "--examples_path",
        default="data/prompt_examples.jsonl",
        help="Original generated examples. Used to recover relation_type if raw rows do not include metadata.",
    )
    p.add_argument("--output_csv", required=True, help="Where to write relation-breakdown CSV.")
    p.add_argument("--figure_dir", default=None, help="Optional directory for PNG/PDF figures.")
    p.add_argument("--figure_prefix", default="factual_association_specs25", help="Prefix for figure filenames.")
    p.add_argument("--no_pdf", action="store_true", help="Write PNG only, not PDF copies.")
    p.add_argument(
        "--include_non_factual",
        action="store_true",
        help="By default, non-factual rows are ignored. Set this to include all rows.",
    )
    return p.parse_args()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_examples_by_id(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        print(f"Warning: examples_path not found: {path}. Proceeding without join.")
        return {}
    return {row["example_id"]: row for row in read_jsonl(path)}


def infer_condition(filename: str) -> str:
    name = Path(filename).name
    if name.startswith("baseline"):
        return "baseline"
    if name.startswith("adapter"):
        return "adapter"
    return "unknown"


def infer_run_metadata(filename: str) -> Dict[str, Any]:
    name = Path(filename).name

    m_specs = re.search(r"specs(\d+)", name)
    m_budget = re.search(r"budget(\d+)", name)
    m_lt = re.search(r"calib_(.*?)_specs\d+_budget\d+", name)

    return {
        "condition": infer_condition(name),
        "n_specs": int(m_specs.group(1)) if m_specs else None,
        "budget": int(m_budget.group(1)) if m_budget else None,
        "filename_learning_type": m_lt.group(1) if m_lt else None,
    }


def bool_int(row: Dict[str, Any], key: str, fallback: str | None = None) -> int:
    if key in row:
        return int(bool(row[key]))
    if fallback and fallback in row:
        return int(bool(row[fallback]))
    return 0


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def first_present(*values: Any, default: str = "unknown") -> str:
    for val in values:
        if val not in (None, ""):
            return str(val)
    return default


def get_scoring_type(row: Dict[str, Any], ex: Dict[str, Any]) -> str:
    if row.get("scoring_type"):
        return str(row["scoring_type"])

    row_scoring = row.get("scoring", {})
    if isinstance(row_scoring, dict) and row_scoring.get("scoring_type"):
        return str(row_scoring["scoring_type"])

    ex_scoring = ex.get("scoring", {}) if isinstance(ex, dict) else {}
    if isinstance(ex_scoring, dict) and ex_scoring.get("scoring_type"):
        return str(ex_scoring["scoring_type"])

    return "unknown"


def get_relation_type(row: Dict[str, Any], ex: Dict[str, Any]) -> str:
    """
    Recover relation type from whichever location has it.

    Preferred:
      row["metadata"]["relation_type"]
      row["latent_spec"]["relation"]
      original_example["metadata"]["relation_type"]
      original_example["latent_spec"]["relation"]
    """
    row_meta = row.get("metadata", {})
    if isinstance(row_meta, dict) and row_meta.get("relation_type"):
        return str(row_meta["relation_type"])

    row_latent = row.get("latent_spec", {})
    if isinstance(row_latent, dict) and row_latent.get("relation"):
        return str(row_latent["relation"])

    ex_meta = ex.get("metadata", {}) if isinstance(ex, dict) else {}
    if isinstance(ex_meta, dict) and ex_meta.get("relation_type"):
        return str(ex_meta["relation_type"])

    ex_latent = ex.get("latent_spec", {}) if isinstance(ex, dict) else {}
    if isinstance(ex_latent, dict) and ex_latent.get("relation"):
        return str(ex_latent["relation"])

    return "unknown"


def get_example_mode(row: Dict[str, Any], ex: Dict[str, Any]) -> str:
    if row.get("example_mode"):
        return str(row["example_mode"])

    row_meta = row.get("metadata", {})
    if isinstance(row_meta, dict) and row_meta.get("example_mode"):
        return str(row_meta["example_mode"])

    ex_meta = ex.get("metadata", {}) if isinstance(ex, dict) else {}
    if isinstance(ex_meta, dict) and ex_meta.get("example_mode"):
        return str(ex_meta["example_mode"])

    return "unknown"


def add(summary: Dict[Tuple[str, ...], Counter], key: Tuple[str, ...], row: Dict[str, Any]) -> None:
    summary[key]["n"] += 1

    summary[key]["loose_correct"] += bool_int(row, "loose_score", fallback="passed")
    summary[key]["strict_correct"] += bool_int(row, "strict_score", fallback="passed")
    summary[key]["concept_correct"] += bool_int(row, "concept_score", fallback="target_mentioned")
    summary[key]["passed_correct"] += bool_int(row, "passed", fallback="strict_score")

    if row.get("target_mentioned"):
        summary[key]["target_mentioned"] += 1
    if row.get("contains_exact"):
        summary[key]["contains_exact"] += 1
    if row.get("starts_with_rejection"):
        summary[key]["starts_with_rejection"] += 1
    if row.get("has_affirmation"):
        summary[key]["has_affirmation"] += 1


def write_summary(args: argparse.Namespace) -> Path:
    examples_by_id = load_examples_by_id(args.examples_path)
    summary: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    for input_path in args.inputs:
        meta = infer_run_metadata(input_path)
        rows = read_jsonl(input_path)

        for row in rows:
            ex = examples_by_id.get(row.get("example_id"), {})

            learning_type = first_present(
                row.get("learning_type"),
                ex.get("learning_type") if isinstance(ex, dict) else None,
                meta["filename_learning_type"],
            )

            if learning_type != "factual_association" and not args.include_non_factual:
                continue

            split = first_present(row.get("split"), ex.get("split") if isinstance(ex, dict) else None)
            mode = get_example_mode(row, ex)
            scoring_type = get_scoring_type(row, ex)
            relation_type = get_relation_type(row, ex)

            # Specific relation row.
            key = (
                Path(input_path).name,
                meta["condition"],
                str(meta["n_specs"]),
                str(meta["budget"]),
                learning_type,
                relation_type,
                split,
                mode,
                scoring_type,
            )
            add(summary, key, row)

            # ALL_RELATIONS aggregate row for easier comparison.
            key_all_rel = (
                Path(input_path).name,
                meta["condition"],
                str(meta["n_specs"]),
                str(meta["budget"]),
                learning_type,
                "ALL_RELATIONS",
                split,
                mode,
                scoring_type,
            )
            add(summary, key_all_rel, row)

            # ALL_MODES aggregate for split-level relation breakdown.
            key_all_modes = (
                Path(input_path).name,
                meta["condition"],
                str(meta["n_specs"]),
                str(meta["budget"]),
                learning_type,
                relation_type,
                split,
                "ALL_MODES",
                "ALL_SCORING_TYPES",
            )
            add(summary, key_all_modes, row)

            # ALL_RELATIONS + ALL_MODES aggregate.
            key_all_rel_modes = (
                Path(input_path).name,
                meta["condition"],
                str(meta["n_specs"]),
                str(meta["budget"]),
                learning_type,
                "ALL_RELATIONS",
                split,
                "ALL_MODES",
                "ALL_SCORING_TYPES",
            )
            add(summary, key_all_rel_modes, row)

    fields = [
        "file",
        "condition",
        "n_specs",
        "budget",
        "learning_type",
        "relation_type",
        "split",
        "example_mode",
        "scoring_type",
        "n",
        "loose_correct",
        "loose_accuracy",
        "strict_correct",
        "strict_accuracy",
        "concept_correct",
        "concept_accuracy",
        "passed_correct",
        "passed_accuracy",
        "target_mentioned_rate",
        "contains_exact_rate",
        "starts_with_rejection_rate",
        "affirmation_rate",
    ]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for key, c in sorted(summary.items()):
            (
                file,
                condition,
                n_specs,
                budget,
                learning_type,
                relation_type,
                split,
                mode,
                scoring_type,
            ) = key
            n = c["n"]

            w.writerow({
                "file": file,
                "condition": condition,
                "n_specs": n_specs,
                "budget": budget,
                "learning_type": learning_type,
                "relation_type": relation_type,
                "split": split,
                "example_mode": mode,
                "scoring_type": scoring_type,
                "n": n,
                "loose_correct": c["loose_correct"],
                "loose_accuracy": ratio(c["loose_correct"], n),
                "strict_correct": c["strict_correct"],
                "strict_accuracy": ratio(c["strict_correct"], n),
                "concept_correct": c["concept_correct"],
                "concept_accuracy": ratio(c["concept_correct"], n),
                "passed_correct": c["passed_correct"],
                "passed_accuracy": ratio(c["passed_correct"], n),
                "target_mentioned_rate": ratio(c["target_mentioned"], n),
                "contains_exact_rate": ratio(c["contains_exact"], n),
                "starts_with_rejection_rate": ratio(c["starts_with_rejection"], n),
                "affirmation_rate": ratio(c["has_affirmation"], n),
            })

    print(f"Wrote {out_path}")
    return out_path


def save_figure(fig, figure_dir: Path, stem: str, no_pdf: bool) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {png_path}")

    if not no_pdf:
        pdf_path = figure_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

    plt.close(fig)


def read_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in [
        "n_specs",
        "budget",
        "n",
        "loose_accuracy",
        "strict_accuracy",
        "concept_accuracy",
        "target_mentioned_rate",
        "starts_with_rejection_rate",
        "affirmation_rate",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_relation_split_budget(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    """
    Figure 1: one panel per split, relation-type strict accuracy across budgets.
    Uses adapter rows and ALL_MODES aggregates.
    """
    sub = df[
        (df["condition"] == "adapter")
        & (df["relation_type"].isin(RELATION_ORDER))
        & (df["example_mode"] == "ALL_MODES")
        & (df["scoring_type"] == "ALL_SCORING_TYPES")
    ].copy()

    if sub.empty:
        print("Skipping relation split-budget figure: no matching rows.")
        return

    for split in SPLIT_ORDER:
        split_df = sub[sub["split"] == split].copy()
        if split_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for rel in RELATION_ORDER:
            rel_df = split_df[split_df["relation_type"] == rel].sort_values("budget")
            if rel_df.empty:
                continue
            ax.plot(
                rel_df["budget"],
                rel_df["strict_accuracy"],
                marker="o",
                label=RELATION_LABELS.get(rel, rel),
            )

        budgets = sorted([int(x) for x in split_df["budget"].dropna().unique()])
        if budgets:
            ax.set_xticks(budgets)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training budget per latent specification")
        ax.set_ylabel("Strict accuracy")
        ax.set_title(f"Factual association by relation: {SPLIT_LABELS.get(split, split)}")
        ax.legend(frameon=True)
        fig.tight_layout()

        save_figure(fig, figure_dir, f"{prefix}_relation_breakdown_{split}", no_pdf)


def plot_relation_mode_budget(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    """
    Figure 2: one panel per mode, relation-type strict accuracy across budgets.
    Uses adapter rows and mode-specific entries.
    """
    sub = df[
        (df["condition"] == "adapter")
        & (df["relation_type"].isin(RELATION_ORDER))
        & (df["example_mode"].isin(MODE_ORDER))
    ].copy()

    if sub.empty:
        print("Skipping relation mode-budget figure: no matching rows.")
        return

    for mode in MODE_ORDER:
        mode_df = sub[sub["example_mode"] == mode].copy()
        if mode_df.empty:
            continue

        # Average across duplicate split/scoring rows for readability.
        mode_df = (
            mode_df.groupby(["budget", "relation_type"], as_index=False)["strict_accuracy"]
            .mean()
            .sort_values(["relation_type", "budget"])
        )

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for rel in RELATION_ORDER:
            rel_df = mode_df[mode_df["relation_type"] == rel].sort_values("budget")
            if rel_df.empty:
                continue
            ax.plot(
                rel_df["budget"],
                rel_df["strict_accuracy"],
                marker="o",
                label=RELATION_LABELS.get(rel, rel),
            )

        budgets = sorted([int(x) for x in mode_df["budget"].dropna().unique()])
        if budgets:
            ax.set_xticks(budgets)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training budget per latent specification")
        ax.set_ylabel("Strict accuracy")
        ax.set_title(f"Factual association by relation: {MODE_LABELS.get(mode, mode)}")
        ax.legend(frameon=True)
        fig.tight_layout()

        save_figure(fig, figure_dir, f"{prefix}_relation_breakdown_mode_{mode}", no_pdf)


def plot_budget12_relation_bars(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    """
    Figure 3: budget12 relation-type bars for the major split-level outcomes.
    Useful for quickly seeing which relation is dragging performance down.
    """
    sub = df[
        (df["condition"] == "adapter")
        & (df["relation_type"].isin(RELATION_ORDER))
        & (df["example_mode"] == "ALL_MODES")
        & (df["scoring_type"] == "ALL_SCORING_TYPES")
    ].copy()

    if sub.empty:
        print("Skipping budget12 bars: no matching rows.")
        return

    max_budget = int(sub["budget"].max())
    sub = sub[sub["budget"] == max_budget]

    for split in SPLIT_ORDER:
        split_df = sub[sub["split"] == split].copy()
        if split_df.empty:
            continue

        split_df["relation_label"] = split_df["relation_type"].map(RELATION_LABELS).fillna(split_df["relation_type"])
        split_df = split_df.set_index("relation_label").reindex([RELATION_LABELS[r] for r in RELATION_ORDER]).reset_index()

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.bar(split_df["relation_label"], split_df["strict_accuracy"])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Relation type")
        ax.set_ylabel("Strict accuracy")
        ax.set_title(f"Factual relation breakdown at budget {max_budget}: {SPLIT_LABELS.get(split, split)}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()

        save_figure(fig, figure_dir, f"{prefix}_budget{max_budget}_bars_{split}", no_pdf)


def make_figures(summary_csv: Path, figure_dir: str | Path, prefix: str, no_pdf: bool) -> None:
    df = read_summary_csv(summary_csv)
    figure_dir = Path(figure_dir)

    plot_relation_split_budget(df, figure_dir, prefix, no_pdf)
    plot_relation_mode_budget(df, figure_dir, prefix, no_pdf)
    plot_budget12_relation_bars(df, figure_dir, prefix, no_pdf)


def main() -> None:
    args = parse_args()
    out_csv = write_summary(args)

    if args.figure_dir:
        make_figures(out_csv, args.figure_dir, args.figure_prefix, args.no_pdf)


if __name__ == "__main__":
    main()
