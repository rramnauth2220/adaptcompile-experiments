#!/usr/bin/env python3
"""Build the per-objective Experiment 4 representation-ablation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from plot_exp4_ablation import FEATURE_DIRS, _load_one, validate_sources
from plot_style import (
    OBJECTIVE_LABELS,
    OBJECTIVE_ORDER,
    OUTCOMES,
    json_script_metadata,
    prediction_selected_utilities,
    to_builtin,
)


FEATURE_COLUMNS: Sequence[Tuple[str, str]] = (
    ("ablation_episode", "Episode"),
    ("full_auto", "Full"),
    ("ablation_module_probes", "Probe"),
    ("ablation_frozen_behavior", "Frozen"),
)
TABLE_COLUMNS = [
    "Objective",
    "Episode MAE",
    "Full MAE",
    "Probe MAE",
    "Frozen MAE",
    "Episode Regret",
    "Full Regret",
    "Probe Regret",
    "Frozen Regret",
]


def _prediction_mae_by_objective(predictions: pd.DataFrame) -> Mapping[str, float]:
    missing = []
    for outcome in OUTCOMES:
        missing.extend(
            column
            for column in [f"observed_{outcome}", f"primary_predicted_{outcome}"]
            if column not in predictions.columns
        )
    if missing:
        raise ValueError(f"Prediction table missing columns: {sorted(set(missing))}")

    abs_errors = [
        (predictions[f"observed_{outcome}"].astype(float) - predictions[f"primary_predicted_{outcome}"].astype(float)).abs()
        for outcome in OUTCOMES
    ]
    work = predictions.copy()
    work["_geometry_mae"] = sum(abs_errors) / len(abs_errors)
    return work.groupby("learning_type")["_geometry_mae"].mean().astype(float).to_dict()


def _regret_by_objective(predictions: pd.DataFrame) -> Mapping[str, float]:
    selected = prediction_selected_utilities(predictions, method_prefix="primary")
    return selected.groupby("learning_type")["oracle_regret"].mean().astype(float).to_dict()


def collect_table(experiment2_root: Path, sanity_tolerance: float) -> Tuple[List[Dict[str, Any]], List[Path], Dict[str, Any]]:
    loaded = [_load_one(experiment2_root, feature_dir) for feature_dir in FEATURE_DIRS]
    checks = validate_sources(loaded, sanity_tolerance=sanity_tolerance)
    by_dir = {str(item["feature_dir"]): item for item in loaded}

    mae_by_feature = {
        feature_dir: _prediction_mae_by_objective(by_dir[feature_dir]["predictions"]) for feature_dir, _ in FEATURE_COLUMNS
    }
    regret_by_feature = {
        feature_dir: _regret_by_objective(by_dir[feature_dir]["predictions"]) for feature_dir, _ in FEATURE_COLUMNS
    }

    rows: List[Dict[str, Any]] = []
    for objective in OBJECTIVE_ORDER:
        row: Dict[str, Any] = {"Objective": OBJECTIVE_LABELS.get(objective, objective)}
        for feature_dir, prefix in FEATURE_COLUMNS:
            row[f"{prefix} MAE"] = float(mae_by_feature[feature_dir][objective])
        for feature_dir, prefix in FEATURE_COLUMNS:
            row[f"{prefix} Regret"] = float(regret_by_feature[feature_dir][objective])
        rows.append(row)

    inputs: List[Path] = []
    for item in loaded:
        inputs.extend([item["predictions_path"], item["metrics_path"], item["ranking_path"], item["metadata_path"]])
    return rows, inputs, checks


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], digits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(TABLE_COLUMNS) - 1)) + " |",
    ]
    for row in rows:
        values = [str(row["Objective"])]
        values.extend(f"{float(row[column]):.{digits}f}" for column in TABLE_COLUMNS[1:])
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[Path],
    checks: Mapping[str, Any],
    output_csv: Path,
    output_md: Path,
    digits: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TABLE_COLUMNS).to_csv(output_csv, index=False)
    _write_markdown(output_md, rows, digits=digits)

    metadata = json_script_metadata(__file__, inputs)
    metadata.update(
        {
            "table": "exp4_ablation_by_objective",
            "output_csv": str(output_csv),
            "output_markdown": str(output_md),
            "columns": TABLE_COLUMNS,
            "metric_definitions": {
                "MAE": "Mean absolute error over acquisition, transfer, boundedness, and preservation for held-out test episode-program rows within the objective.",
                "Regret": "Mean held-out episode oracle regret from the tie-aware primary predicted-utility selector.",
            },
            "data_integrity_checks": checks,
        }
    )
    output_csv.with_suffix(".metadata.json").write_text(json.dumps(to_builtin(metadata), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the per-objective Experiment 4 ablation table.")
    parser.add_argument("--experiment2_root", default="artifacts/llama")
    parser.add_argument("--output_csv", default="outputs/release_verification/tables/exp4_ablation_by_objective.csv")
    parser.add_argument("--output_md", default="outputs/release_verification/tables/exp4_ablation_by_objective.md")
    parser.add_argument("--digits", type=int, default=4, help="Decimal places for the Markdown table.")
    parser.add_argument(
        "--sanity_tolerance",
        type=float,
        default=0.002,
        help="Maximum allowed absolute deviation from recorded overall Exp 4 sanity values.",
    )
    args = parser.parse_args()

    rows, inputs, checks = collect_table(Path(args.experiment2_root), sanity_tolerance=args.sanity_tolerance)
    write_outputs(rows, inputs, checks, Path(args.output_csv), Path(args.output_md), digits=args.digits)
    print(pd.DataFrame(rows, columns=TABLE_COLUMNS).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
