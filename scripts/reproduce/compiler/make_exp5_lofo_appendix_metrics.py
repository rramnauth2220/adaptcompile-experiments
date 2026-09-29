#!/usr/bin/env python3
"""Populate the Experiment 5 LOFO appendix metrics table from saved outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from plot_style import OBJECTIVE_LABELS, OBJECTIVE_ORDER, PROGRAM_ORDER, git_commit, to_builtin


OUTPUT_COLUMNS = [
    "held_out_family",
    "display_name",
    "validation_mae",
    "test_mae",
    "utility_corr",
    "spearman",
    "pairwise_accuracy",
    "top1",
    "top2",
]
DISPLAY_COLUMNS = [
    "Held-out family",
    "Val. MAE",
    "Test MAE",
    "Utility Corr.",
    "Spearman",
    "Pairwise",
    "Top-1",
    "Top-2",
]
EXPECTED_CHECKS = {
    "behavioral_policy": {
        "validation_mae": 0.04975,
        "test_mae": 0.45056,
        "utility_corr": 0.8623,
        "spearman": 0.520,
        "pairwise_accuracy": 0.7083,
        "top1": 0.30,
        "top2": 0.60,
    },
    "causal_mapping": {
        "test_mae": 0.29541,
        "utility_corr": 0.6457,
        "spearman": 0.470,
        "pairwise_accuracy": 0.6917,
        "top1": 0.10,
    },
    "factual_association": {
        "test_mae": 0.40863,
        "utility_corr": -0.1099,
        "spearman": -0.100,
        "pairwise_accuracy": 0.4833,
        "top1": 0.20,
    },
    "lexical_binding": {
        "test_mae": 0.34126,
        "utility_corr": -0.0262,
        "spearman": -0.550,
        "pairwise_accuracy": 0.2583,
        "top1": 0.05,
    },
    "procedural_reasoning": {
        "test_mae": 0.20002,
        "utility_corr": -0.0373,
        "spearman": -0.050,
        "pairwise_accuracy": 0.4750,
        "top1": 0.05,
    },
    "macro_average": {
        "validation_mae": 0.04188,
        "test_mae": 0.33918,
        "utility_corr": 0.26693,
        "spearman": 0.058,
        "pairwise_accuracy": 0.52333,
        "top1": 0.14,
        "top2": 0.47,
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except KeyError as exc:
        raise ValueError(f"Missing required field {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value for {key}: {row[key]!r}")
    return value


def assert_close(name: str, observed: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: observed={observed}, expected approximately={expected}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def mean(values: Iterable[float | bool]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        raise ValueError("Cannot compute mean over an empty sequence.")
    return float(np.mean(vals))


def primary_by_episode_metric(path: Path, metric: str) -> float:
    rows = [row for row in read_csv_rows(path) if str(row.get("method")) == "primary"]
    if not rows:
        raise ValueError(f"{path} contains no primary ranking rows.")
    if metric in {"top1_set_oracle_recovery", "top2_set_oracle_recovery", "selected_config_oracle_optimal"}:
        return mean(parse_bool(row[metric]) for row in rows)
    return mean(float(row[metric]) for row in rows)


def validate_fold(
    lofo_root: Path,
    held_out: str,
    summary_row: Mapping[str, Any],
    source_files: List[Path],
) -> None:
    fold_root = lofo_root / held_out
    metadata_path = fold_root / "model_metadata.json"
    prediction_metrics_path = fold_root / "prediction_metrics.json"
    ranking_metrics_path = fold_root / "ranking_metrics.json"
    ranking_by_episode_path = fold_root / "ranking_metrics_by_episode.csv"
    for path in [metadata_path, prediction_metrics_path, ranking_metrics_path, ranking_by_episode_path]:
        if not path.exists():
            raise FileNotFoundError(path)
        source_files.append(path)

    metadata = read_json(metadata_path)
    if metadata.get("experiment") != "experiment2_lofo_generalization":
        raise ValueError(f"{metadata_path} is not a final LOFO evaluator metadata file.")
    if metadata.get("feature_set") != "episode":
        raise ValueError(f"{metadata_path} does not use the final episode-only representation.")
    if metadata.get("held_out_learning_type") != held_out:
        raise ValueError(f"{metadata_path} held_out_learning_type mismatch.")
    if metadata.get("objective_identity_used_in_primary_predictor") is not False:
        raise ValueError(f"{metadata_path} indicates objective identity was used.")
    if metadata.get("held_out_objective_seen_during_training") is not False:
        raise ValueError(f"{metadata_path} indicates held-out objective was seen during training.")
    if metadata.get("held_out_objective_seen_during_validation") is not False:
        raise ValueError(f"{metadata_path} indicates held-out objective was seen during validation.")
    if metadata.get("test_rows_used_for_model_fitting_or_selection") is not False:
        raise ValueError(f"{metadata_path} indicates test rows were used for fitting or selection.")
    if held_out in set(metadata.get("training_learning_types", [])):
        raise ValueError(f"{held_out} appears in LOFO training types.")
    if held_out in set(metadata.get("validation_learning_types", [])):
        raise ValueError(f"{held_out} appears in LOFO validation types.")
    if set(metadata.get("test_learning_types", [])) != {held_out}:
        raise ValueError(f"{metadata_path} test_learning_types are not exactly [{held_out!r}].")
    if list(metadata.get("config_ids", [])) != list(PROGRAM_ORDER):
        raise ValueError(f"{metadata_path} does not use the primary budget-matched config library.")
    if int(metadata.get("validation_episodes", -1)) != 80 or int(metadata.get("test_episodes", -1)) != 20:
        raise ValueError(f"{metadata_path} has unexpected LOFO validation/test episode counts.")
    assert_close(
        f"{held_out} metadata validation MAE",
        float(metadata["validation_mean_mae"]),
        as_float(summary_row, "validation_mean_mae"),
        tolerance=1e-12,
    )

    prediction_metrics = read_json(prediction_metrics_path)
    ranking_metrics = read_json(ranking_metrics_path)
    primary_prediction = prediction_metrics["primary"]
    primary_ranking = ranking_metrics["primary"]

    exact_pairs = {
        "test_mean_mae": float(primary_prediction["mean_mae"]),
        "utility_correlation": float(prediction_metrics["utility_correlation"]["primary"]),
        "mean_episode_spearman": float(primary_ranking["mean_episode_spearman"]),
        "pairwise_ranking_accuracy": float(primary_ranking["pairwise_ranking_accuracy"]),
        "top1_oracle_recovery": float(primary_ranking["top1_set_oracle_recovery"]),
        "top2_oracle_recovery": float(primary_ranking["top2_set_oracle_recovery"]),
    }
    for summary_key, expected_value in exact_pairs.items():
        assert_close(
            f"{held_out} {summary_key} metric-file cross-check",
            as_float(summary_row, summary_key),
            expected_value,
            tolerance=1e-12,
        )

    by_episode_pairs = {
        "mean_episode_spearman": primary_by_episode_metric(ranking_by_episode_path, "spearman"),
        "pairwise_ranking_accuracy": primary_by_episode_metric(ranking_by_episode_path, "pairwise_ranking_accuracy"),
        "top1_oracle_recovery": primary_by_episode_metric(ranking_by_episode_path, "top1_set_oracle_recovery"),
        "top2_oracle_recovery": primary_by_episode_metric(ranking_by_episode_path, "top2_set_oracle_recovery"),
    }
    for summary_key, expected_value in by_episode_pairs.items():
        assert_close(
            f"{held_out} {summary_key} per-episode cross-check",
            as_float(summary_row, summary_key),
            expected_value,
            tolerance=1e-12,
        )


def output_row(summary_row: Mapping[str, Any]) -> Dict[str, Any]:
    held_out = str(summary_row["held_out_learning_type"])
    return {
        "held_out_family": held_out,
        "display_name": OBJECTIVE_LABELS.get(held_out, "Macro"),
        "validation_mae": as_float(summary_row, "validation_mean_mae"),
        "test_mae": as_float(summary_row, "test_mean_mae"),
        "utility_corr": as_float(summary_row, "utility_correlation"),
        "spearman": as_float(summary_row, "mean_episode_spearman"),
        "pairwise_accuracy": as_float(summary_row, "pairwise_ranking_accuracy"),
        "top1": as_float(summary_row, "top1_oracle_recovery"),
        "top2": as_float(summary_row, "top2_oracle_recovery"),
    }


def collect_rows(lofo_root: Path, tolerance: float) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Path]]:
    summary_csv = lofo_root / "lofo_summary.csv"
    summary_json = lofo_root / "lofo_summary.json"
    source_files = [summary_csv, summary_json]
    rows = read_csv_rows(summary_csv)
    summary_payload = read_json(summary_json)
    if summary_payload.get("experiment") != "experiment2_lofo_generalization":
        raise ValueError(f"{summary_json} is not the final LOFO evaluator summary.")
    if summary_payload.get("feature_set") != "episode":
        raise ValueError(f"{summary_json} is not the final episode-only LOFO run.")
    if list(summary_payload.get("config_ids", [])) != list(PROGRAM_ORDER):
        raise ValueError(f"{summary_json} does not use the primary budget-matched config library.")

    expected_order = [*OBJECTIVE_ORDER, "macro_average"]
    by_family = {str(row["held_out_learning_type"]): row for row in rows}
    missing = [family for family in expected_order if family not in by_family]
    if missing:
        raise ValueError(f"LOFO summary missing expected rows: {missing}")
    extra = sorted(set(by_family) - set(expected_order))
    if extra:
        raise ValueError(f"LOFO summary contains unexpected rows: {extra}")

    for held_out in OBJECTIVE_ORDER:
        validate_fold(lofo_root, held_out, by_family[held_out], source_files)

    out = [output_row(by_family[family]) for family in expected_order]
    fold_rows = out[:-1]
    macro = out[-1]
    macro_checks = {
        "validation_mae": mean(row["validation_mae"] for row in fold_rows),
        "test_mae": mean(row["test_mae"] for row in fold_rows),
        "utility_corr": mean(row["utility_corr"] for row in fold_rows),
        "spearman": mean(row["spearman"] for row in fold_rows),
        "pairwise_accuracy": mean(row["pairwise_accuracy"] for row in fold_rows),
        "top1": mean(row["top1"] for row in fold_rows),
        "top2": mean(row["top2"] for row in fold_rows),
    }
    for key, computed in macro_checks.items():
        assert_close(f"macro {key}", macro[key], computed, tolerance=1e-12)

    for family, expected in EXPECTED_CHECKS.items():
        row = next(item for item in out if item["held_out_family"] == family)
        for key, expected_value in expected.items():
            assert_close(f"{family} sanity {key}", row[key], expected_value, tolerance=tolerance)

    checks = {
        "final_lofo_run": str(lofo_root),
        "feature_set": "episode",
        "held_out_objective_absent_from_training_and_validation": True,
        "macro_validation_mae_from_folds": macro_checks["validation_mae"],
        "macro_top2_from_folds": macro_checks["top2"],
        "sanity_tolerance": tolerance,
    }
    return out, checks, source_files


def write_output(path: Path, rows: Sequence[Mapping[str, Any]], checks: Mapping[str, Any], source_files: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})
    metadata = {
        "script": __file__,
        "git_commit": git_commit(),
        "output_csv": str(path),
        "source_files": [str(path) for path in source_files],
        "data_integrity_checks": dict(checks),
        "columns": OUTPUT_COLUMNS,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(to_builtin(metadata), indent=2), encoding="utf-8")


def print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                str(row["display_name"]),
                f"{float(row['validation_mae']):.3f}",
                f"{float(row['test_mae']):.3f}",
                f"{float(row['utility_corr']):.3f}",
                f"{float(row['spearman']):.3f}",
                f"{float(row['pairwise_accuracy']):.3f}",
                f"{float(row['top1']):.2f}",
                f"{float(row['top2']):.2f}",
            ]
        )
    widths = [
        max(len(str(value)) for value in [header, *[row[i] for row in display_rows]])
        for i, header in enumerate(DISPLAY_COLUMNS)
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(DISPLAY_COLUMNS)))
    print("  ".join("-" * width for width in widths))
    for row in display_rows:
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Experiment 5 LOFO appendix metrics table.")
    parser.add_argument("--lofo_root", default="artifacts/llama/lofo")
    parser.add_argument("--output_csv", default="outputs/release_verification/tables/lofo_appendix_metrics.csv")
    parser.add_argument("--sanity_tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    rows, checks, source_files = collect_rows(Path(args.lofo_root), tolerance=args.sanity_tolerance)
    write_output(Path(args.output_csv), rows, checks, source_files)
    print_table(rows)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
