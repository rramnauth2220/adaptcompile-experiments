#!/usr/bin/env python3
"""
Summarize localization-result JSONLs.

The output groups by:
  learning_type, budget, seed, localization_condition, split, example_mode, scoring_type

It also writes ALL_MODES aggregate rows per split.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def expand_inputs(inputs: List[str]) -> List[str]:
    paths = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            paths.extend(matches)
        else:
            paths.append(item)
    return sorted(set(paths))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bool_int(row: Dict[str, Any], key: str, fallback: str | None = None) -> int:
    if key in row:
        return int(bool(row[key]))
    if fallback and fallback in row:
        return int(bool(row[fallback]))
    return 0


def infer_budget(row: Dict[str, Any], path: str | Path) -> int | None:
    if row.get("train_budget_per_spec") is not None:
        return int(row["train_budget_per_spec"])
    import re
    m = re.search(r"budget(\d+)", str(path))
    return int(m.group(1)) if m else None


def infer_seed(row: Dict[str, Any], path: str | Path) -> int | None:
    if row.get("seed") is not None:
        return int(row["seed"])
    import re
    m = re.search(r"seed(\d+)", str(path))
    return int(m.group(1)) if m else None


def row_record(row: Dict[str, Any], path: str | Path) -> Dict[str, Any]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
    scoring = row.get("scoring", {}) if isinstance(row.get("scoring", {}), dict) else {}

    return {
        "file": Path(path).name,
        "learning_type": row.get("learning_type", "unknown"),
        "budget": infer_budget(row, path),
        "seed": infer_seed(row, path),
        "condition": row.get("condition", row.get("localization_condition", "unknown")),
        "localization_condition": row.get("localization_condition", row.get("condition", "unknown")),
        "split": row.get("split", "unknown"),
        "example_mode": row.get("example_mode", metadata.get("example_mode", "unknown")),
        "scoring_type": row.get("scoring_type", scoring.get("scoring_type", "unknown")),
        "loose_score": bool_int(row, "loose_score", fallback="passed"),
        "strict_score": bool_int(row, "strict_score", fallback="passed"),
        "concept_score": bool_int(row, "concept_score", fallback="target_mentioned"),
        "passed": bool_int(row, "passed", fallback="strict_score"),
        "starts_with_rejection": bool_int(row, "starts_with_rejection"),
        "affirmation": bool_int(row, "affirmation"),
        "target_mentioned": bool_int(row, "target_mentioned"),
        "contains_exact": bool_int(row, "contains_exact"),
    }


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        out = dict(zip(group_cols, key))
        n = len(g)
        out.update({
            "n": n,
            "loose_accuracy": g["loose_score"].mean(),
            "strict_accuracy": g["strict_score"].mean(),
            "concept_accuracy": g["concept_score"].mean(),
            "passed_accuracy": g["passed"].mean(),
            "starts_with_rejection_rate": g["starts_with_rejection"].mean(),
            "affirmation_rate": g["affirmation"].mean(),
            "target_mentioned_rate": g["target_mentioned"].mean(),
            "contains_exact_rate": g["contains_exact"].mean(),
        })
        rows.append(out)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)

    records = []
    for path in paths:
        for row in read_jsonl(path):
            records.append(row_record(row, path))

    df = pd.DataFrame(records)

    base_cols = [
        "learning_type",
        "budget",
        "seed",
        "localization_condition",
        "split",
        "example_mode",
        "scoring_type",
    ]

    exact = summarize_group(df, base_cols)

    # Add useful ALL_MODES aggregate per split.
    all_modes_df = df.copy()
    all_modes_df["example_mode"] = "ALL_MODES"
    all_modes_df["scoring_type"] = "ALL_SCORING_TYPES"
    all_modes = summarize_group(all_modes_df, base_cols)

    out = pd.concat([exact, all_modes], ignore_index=True).sort_values(
        ["learning_type", "budget", "seed", "localization_condition", "split", "example_mode", "scoring_type"]
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
