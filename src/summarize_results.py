#!/usr/bin/env python3
"""
Summarize mode-aware calibration JSONL outputs and optionally make figures.

Expected input rows are produced by evaluate_calibration_run.py / evaluate_model.py
and should include:
  learning_type, split, loose_score, strict_score

If present, this script also uses:
  example_mode
  scoring_type
  concept_score
  target_mentioned
  starts_with_rejection
  has_affirmation

Outputs:
  1. Summary CSV grouped by:
       file x condition x n_specs x budget x learning_type x split x example_mode x scoring_type
  2. Optional aggregate ALL_MODES rows per split.
  3. Optional appendix-style figures when --figure_dir is provided.

Examples:
  python src/summarize_results.py \
    --inputs outputs/mode_aware_budget_sweep/*.jsonl \
    --output_csv outputs/mode_aware_budget_sweep/summary_lexical_binding_specs25_all_budgets.csv \
    --figure_dir outputs/mode_aware_budget_sweep/figures

The figure code uses matplotlib defaults only: no seaborn and no manually specified colors.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .common import read_jsonl
except ImportError:  # pragma: no cover - script execution path
    from common import read_jsonl


SPLIT_ORDER = ["id_eval", "paraphrase_eval", "generalization", "negative_control"]

SPLIT_LABELS = {
    "id_eval": "ID eval",
    "paraphrase_eval": "Paraphrase",
    "generalization": "Generalization",
    "negative_control": "Negative control",
}

MODE_ORDER = [
    "retrieval",
    "positive_polarity",
    "negative_polarity",
    "generalization_concept",
    "generalization_yes_no",
]

MODE_LABELS = {
    "retrieval": "Retrieval",
    "positive_polarity": "Positive polarity",
    "negative_polarity": "Negative polarity",
    "generalization_concept": "Generalization concept",
    "generalization_yes_no": "Generalization yes/no",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Evaluation JSONL files.")
    p.add_argument("--output_csv", required=True, help="Where to write summary CSV.")
    p.add_argument(
        "--figure_dir",
        default=None,
        help="Optional directory for appendix-style PNG/PDF figures.",
    )
    p.add_argument(
        "--figure_prefix",
        default=None,
        help="Optional prefix for figure filenames. Defaults to inferred learning type/specs.",
    )
    p.add_argument(
        "--no_all_modes",
        action="store_true",
        help="Disable aggregate ALL_MODES rows.",
    )
    p.add_argument(
        "--no_pdf",
        action="store_true",
        help="Only write PNG figures, not PDF copies.",
    )
    return p.parse_args()


def infer_condition(filename: str) -> str:
    name = Path(filename).name
    if name.startswith("baseline"):
        return "baseline"
    if name.startswith("adapter"):
        return "adapter"
    return "unknown"


def infer_run_metadata(filename: str) -> Dict[str, Any]:
    name = Path(filename).name

    specs = None
    budget = None
    learning_type = None

    m_specs = re.search(r"specs(\d+)", name)
    if m_specs:
        specs = int(m_specs.group(1))

    m_budget = re.search(r"budget(\d+)", name)
    if m_budget:
        budget = int(m_budget.group(1))

    # Common filenames:
    # adapter_calib_lexical_binding_specs25_budget8.jsonl
    # baseline_calib_factual_association_specs10_budget4.jsonl
    m_lt = re.search(r"calib_(.*?)_specs\d+_budget\d+", name)
    if m_lt:
        learning_type = m_lt.group(1)

    return {
        "condition": infer_condition(name),
        "filename_learning_type": learning_type,
        "n_specs": specs,
        "budget": budget,
    }


def bool_int(row: Dict[str, Any], key: str, fallback: str | None = None) -> int:
    if key in row:
        return int(bool(row[key]))
    if fallback and fallback in row:
        return int(bool(row[fallback]))
    return 0


def first_present(row: Dict[str, Any], keys: Iterable[str], default: str = "unknown") -> str:
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            return str(val)
    return default


def get_scoring_type(row: Dict[str, Any]) -> str:
    scoring = row.get("scoring_type")
    if scoring:
        return str(scoring)

    scoring_obj = row.get("scoring", {})
    if isinstance(scoring_obj, dict) and scoring_obj.get("scoring_type"):
        return str(scoring_obj["scoring_type"])

    return "unknown"


def add(summary: Dict[Tuple[str, ...], Counter], key: Tuple[str, ...], row: Dict[str, Any]) -> None:
    summary[key]["n"] += 1

    loose = bool_int(row, "loose_score", fallback="passed")
    strict = bool_int(row, "strict_score", fallback="passed")
    concept = bool_int(row, "concept_score", fallback="target_mentioned")
    passed = bool_int(row, "passed", fallback="strict_score")

    summary[key]["loose_correct"] += loose
    summary[key]["strict_correct"] += strict
    summary[key]["concept_correct"] += concept
    summary[key]["passed_correct"] += passed

    if row.get("starts_with_rejection"):
        summary[key]["starts_with_rejection"] += 1
    if row.get("has_affirmation"):
        summary[key]["has_affirmation"] += 1
    if row.get("target_mentioned"):
        summary[key]["target_mentioned"] += 1
    if row.get("contains_exact"):
        summary[key]["contains_exact"] += 1


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def write_summary_csv(args: argparse.Namespace) -> Path:
    summary: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    for path in args.inputs:
        meta = infer_run_metadata(path)
        rows = read_jsonl(path)

        for row in rows:
            file = Path(path).name
            condition = meta["condition"]
            lt = row.get("learning_type") or meta["filename_learning_type"] or "unknown"
            split = row.get("split", "unknown")
            mode = first_present(row, ["example_mode"], default="unknown")
            scoring = get_scoring_type(row)

            base_key = (
                file,
                condition,
                str(meta["n_specs"]),
                str(meta["budget"]),
                lt,
                split,
                mode,
                scoring,
            )
            add(summary, base_key, row)

            if not args.no_all_modes:
                all_modes_key = (
                    file,
                    condition,
                    str(meta["n_specs"]),
                    str(meta["budget"]),
                    lt,
                    split,
                    "ALL_MODES",
                    "ALL_SCORING_TYPES",
                )
                add(summary, all_modes_key, row)

    fields = [
        "file",
        "condition",
        "n_specs",
        "budget",
        "learning_type",
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
        "starts_with_rejection_rate",
        "affirmation_rate",
        "target_mentioned_rate",
        "contains_exact_rate",
    ]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for key, counts in sorted(summary.items()):
            (
                file,
                condition,
                n_specs,
                budget,
                lt,
                split,
                mode,
                scoring,
            ) = key
            n = counts["n"]

            writer.writerow({
                "file": file,
                "condition": condition,
                "n_specs": n_specs,
                "budget": budget,
                "learning_type": lt,
                "split": split,
                "example_mode": mode,
                "scoring_type": scoring,
                "n": n,
                "loose_correct": counts["loose_correct"],
                "loose_accuracy": ratio(counts["loose_correct"], n),
                "strict_correct": counts["strict_correct"],
                "strict_accuracy": ratio(counts["strict_correct"], n),
                "concept_correct": counts["concept_correct"],
                "concept_accuracy": ratio(counts["concept_correct"], n),
                "passed_correct": counts["passed_correct"],
                "passed_accuracy": ratio(counts["passed_correct"], n),
                "starts_with_rejection_rate": ratio(counts["starts_with_rejection"], n),
                "affirmation_rate": ratio(counts["has_affirmation"], n),
                "target_mentioned_rate": ratio(counts["target_mentioned"], n),
                "contains_exact_rate": ratio(counts["contains_exact"], n),
            })

    print(f"Wrote {output_path}")
    return output_path


def save_figure(fig, figure_dir: Path, stem: str, no_pdf: bool) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)

    png_path = figure_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {png_path}")

    if not no_pdf:
        pdf_path = figure_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

    plt.close(fig)


def numeric_summary_df(summary_csv: Path) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(summary_csv)

    for col in [
        "budget",
        "n_specs",
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


def infer_figure_prefix(df: pd.DataFrame, user_prefix: str | None) -> str:
    if user_prefix:
        return user_prefix

    lt_vals = sorted([x for x in df["learning_type"].dropna().unique() if x != "unknown"])
    specs_vals = sorted([int(x) for x in df["n_specs"].dropna().unique()])

    lt = lt_vals[0] if lt_vals else "calibration"
    specs = specs_vals[0] if specs_vals else "unknown"
    return f"{lt}_specs{specs}"


def baseline_by_group(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    import pandas as pd

    base = df[df["condition"] == "baseline"].copy()
    if base.empty:
        return pd.DataFrame(columns=group_cols + [metric])

    return (
        base.sort_values(group_cols + ["budget"])
        .groupby(group_cols, as_index=False)
        .first()[group_cols + [metric]]
    )


def plot_split_budget_summary(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    import matplotlib.pyplot as plt

    """
    Appendix Figure A: split-level strict accuracy across budgets.

    Uses aggregate ALL_MODES rows so the figure mirrors the high-level calibration
    table: ID eval, paraphrase, generalization, negative control.
    """
    sub = df[
        (df["example_mode"] == "ALL_MODES")
        & (df["scoring_type"] == "ALL_SCORING_TYPES")
        & (df["condition"].isin(["baseline", "adapter"]))
    ].copy()

    if sub.empty:
        print("Skipping split budget figure: no ALL_MODES rows found.")
        return

    adapter = sub[sub["condition"] == "adapter"].copy()
    base = baseline_by_group(sub, ["learning_type", "split"], "strict_accuracy")

    if adapter.empty:
        print("Skipping split budget figure: no adapter rows found.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for split in SPLIT_ORDER:
        split_df = adapter[adapter["split"] == split].sort_values("budget")
        if split_df.empty:
            continue

        label = SPLIT_LABELS.get(split, split)
        ax.plot(split_df["budget"], split_df["strict_accuracy"], marker="o", label=f"{label} (adapter)")

        base_match = base[base["split"] == split]
        if not base_match.empty:
            base_val = float(base_match["strict_accuracy"].iloc[0])
            ax.axhline(base_val, linestyle="--", linewidth=1)

    budgets = sorted([int(x) for x in adapter["budget"].dropna().unique()])
    if budgets:
        ax.set_xticks(budgets)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training budget per latent specification")
    ax.set_ylabel("Strict accuracy")
    ax.set_title("Full-stack calibration across budgets")
    ax.legend(frameon=True)
    fig.tight_layout()

    save_figure(fig, figure_dir, f"{prefix}_split_strict_by_budget", no_pdf)


def plot_mode_budget_summary(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    import matplotlib.pyplot as plt

    """
    Appendix Figure B: mode-level strict accuracy across budgets.

    Uses non-ALL_MODES rows to show retrieval, positive polarity,
    negative polarity, and generalization modes separately.
    """
    sub = df[
        (df["condition"] == "adapter")
        & (df["example_mode"] != "ALL_MODES")
    ].copy()

    if sub.empty:
        print("Skipping mode budget figure: no mode-specific adapter rows found.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    plotted = set()
    for mode in MODE_ORDER:
        mode_df = sub[sub["example_mode"] == mode].copy()
        if mode_df.empty:
            continue

        # Some modes may appear across multiple splits. Average across duplicate
        # rows for the same budget/mode to keep the figure readable.
        mode_df = (
            mode_df.groupby(["budget", "example_mode"], as_index=False)["strict_accuracy"]
            .mean()
            .sort_values("budget")
        )

        label = MODE_LABELS.get(mode, mode)
        ax.plot(mode_df["budget"], mode_df["strict_accuracy"], marker="o", label=label)
        plotted.add(mode)

    if not plotted:
        print("Skipping mode budget figure: no recognized modes found.")
        plt.close(fig)
        return

    budgets = sorted([int(x) for x in sub["budget"].dropna().unique()])
    if budgets:
        ax.set_xticks(budgets)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training budget per latent specification")
    ax.set_ylabel("Strict accuracy")
    ax.set_title("Mode-aware calibration across budgets")
    ax.legend(frameon=True)
    fig.tight_layout()

    save_figure(fig, figure_dir, f"{prefix}_mode_strict_by_budget", no_pdf)


def plot_generalization_diagnostic(df: pd.DataFrame, figure_dir: Path, prefix: str, no_pdf: bool) -> None:
    import matplotlib.pyplot as plt

    """
    Appendix Figure C: diagnostic for yes/no generalization.

    Shows loose, strict, and concept accuracy for generalization_yes_no.
    This captures cases where the model gets polarity or concept mention but
    not both at once.
    """
    sub = df[
        (df["condition"] == "adapter")
        & (df["example_mode"] == "generalization_yes_no")
    ].copy()

    if sub.empty:
        print("Skipping generalization diagnostic figure: no generalization_yes_no rows found.")
        return

    # Average duplicate rows if any.
    sub = (
        sub.groupby(["budget"], as_index=False)[
            ["loose_accuracy", "strict_accuracy", "concept_accuracy"]
        ]
        .mean()
        .sort_values("budget")
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.plot(sub["budget"], sub["loose_accuracy"], marker="o", label="Loose accuracy")
    ax.plot(sub["budget"], sub["strict_accuracy"], marker="o", label="Strict accuracy")
    ax.plot(sub["budget"], sub["concept_accuracy"], marker="o", label="Concept accuracy")

    budgets = sorted([int(x) for x in sub["budget"].dropna().unique()])
    if budgets:
        ax.set_xticks(budgets)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training budget per latent specification")
    ax.set_ylabel("Accuracy")
    ax.set_title("Generalization yes/no diagnostic")
    ax.legend(frameon=True)
    fig.tight_layout()

    save_figure(fig, figure_dir, f"{prefix}_generalization_yes_no_diagnostic", no_pdf)


def make_figures(summary_csv: Path, figure_dir: str | Path, figure_prefix: str | None, no_pdf: bool) -> None:
    df = numeric_summary_df(summary_csv)
    figure_dir = Path(figure_dir)
    prefix = infer_figure_prefix(df, figure_prefix)

    plot_split_budget_summary(df, figure_dir, prefix, no_pdf)
    plot_mode_budget_summary(df, figure_dir, prefix, no_pdf)
    plot_generalization_diagnostic(df, figure_dir, prefix, no_pdf)


def main() -> None:
    args = parse_args()
    output_csv = write_summary_csv(args)

    if args.figure_dir:
        make_figures(
            summary_csv=output_csv,
            figure_dir=args.figure_dir,
            figure_prefix=args.figure_prefix,
            no_pdf=args.no_pdf,
        )


if __name__ == "__main__":
    main()
