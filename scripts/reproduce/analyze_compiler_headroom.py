#!/usr/bin/env python3
"""Tie-aware headroom analysis for adaptation-compiler geometry records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compiler_common import read_csv, read_jsonl, write_csv  # noqa: E402


UTILITY_COLUMNS = ["acquisition", "transfer", "boundedness", "preservation"]


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"} or str(path).endswith(".jsonl.gz"):
        return read_jsonl(path)
    return read_csv(path)


def observed_utility(row: Dict[str, Any], utility_columns: Sequence[str]) -> Optional[float]:
    vals = [maybe_float(row.get(col)) for col in utility_columns]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals) / len(vals)


def add_oracle_fields(unit: Dict[str, Any], tie_tolerance: float) -> None:
    utilities = unit["utilities"]
    best = max(utilities.values())
    oracle_set = sorted(cfg for cfg, value in utilities.items() if best - value <= tie_tolerance)
    ordered_values = sorted(utilities.values(), reverse=True)
    top_2_margin = None
    if len(ordered_values) >= 2:
        top_2_margin = ordered_values[0] - ordered_values[1]
    unit["oracle_utility"] = best
    unit["oracle_set"] = oracle_set
    unit["top_2_margin"] = top_2_margin


def build_episode_units(
    records: Sequence[Dict[str, Any]],
    config_ids: Sequence[str] | None,
    utility_columns: Sequence[str],
    tie_tolerance: float,
    allow_incomplete: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    allowed = set(config_ids or [])
    expected = list(config_ids or [])
    buckets: Dict[Tuple[str, int], Dict[str, Any]] = {}
    warnings: List[str] = []

    for row in records:
        config_id = str(row.get("config_id", ""))
        if allowed and config_id not in allowed:
            continue
        utility = observed_utility(row, utility_columns)
        if utility is None:
            continue
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            continue
        seed = int(float(row.get("seed") or 0))
        key = (episode_id, seed)
        bucket = buckets.setdefault(
            key,
            {
                "episode_id": episode_id,
                "seed": seed,
                "meta_split": str(row.get("meta_split", "")),
                "learning_type": str(row.get("learning_type", "")),
                "_utilities": defaultdict(list),
            },
        )
        bucket["_utilities"][config_id].append(float(utility))

    units: List[Dict[str, Any]] = []
    skipped_incomplete = 0
    for bucket in buckets.values():
        utilities = {cfg: float(mean(vals)) for cfg, vals in bucket["_utilities"].items()}
        if expected and not allow_incomplete:
            missing = [cfg for cfg in expected if cfg not in utilities]
            if missing:
                skipped_incomplete += 1
                continue
        if len(utilities) < 2:
            skipped_incomplete += 1
            continue
        unit = {
            "episode_id": bucket["episode_id"],
            "seed": bucket["seed"],
            "meta_split": bucket["meta_split"],
            "learning_type": bucket["learning_type"],
            "utilities": utilities,
        }
        add_oracle_fields(unit, tie_tolerance=tie_tolerance)
        units.append(unit)

    if skipped_incomplete:
        warnings.append(
            f"Skipped {skipped_incomplete} episode/seed units with missing or insufficient config utilities."
        )
    return sorted(units, key=lambda row: (row["meta_split"], row["learning_type"], row["episode_id"], row["seed"])), warnings


def learn_fixed_baseline(
    train_units: Sequence[Dict[str, Any]],
    tie_tolerance: float,
) -> Optional[Dict[str, Any]]:
    by_config: Dict[str, List[float]] = defaultdict(list)
    for unit in train_units:
        for config_id, utility in unit["utilities"].items():
            by_config[config_id].append(float(utility))
    means = {config_id: float(mean(vals)) for config_id, vals in by_config.items()}
    if not means:
        return None
    best = max(means.values())
    tie_set = sorted(config_id for config_id, value in means.items() if best - value <= tie_tolerance)
    return {
        "selected_config_id": tie_set[0],
        "tie_set": tie_set,
        "train_mean_utility": best,
        "train_mean_by_config": means,
        "n_train_units": len(train_units),
    }


def group_units(units: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    groups: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "all", list(units))]
    for learning_type in sorted({unit["learning_type"] for unit in units}):
        groups.append(
            (
                "learning_type",
                learning_type,
                [unit for unit in units if unit["learning_type"] == learning_type],
            )
        )
    return groups


def config_set_id(config_ids: Sequence[str]) -> str:
    return "|".join(sorted(config_ids))


def summarize_winners(units: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    winning_rows: List[Dict[str, Any]] = []
    tie_set_rows: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []

    for group_type, group_value, group in group_units(units):
        n = len(group)
        membership = Counter()
        fractional = Counter()
        unique = Counter()
        tie_sets = Counter()
        margins = [float(unit["top_2_margin"]) for unit in group if unit["top_2_margin"] is not None]

        for unit in group:
            oracle_set = unit["oracle_set"]
            tie_sets[config_set_id(oracle_set)] += 1
            for config_id in oracle_set:
                membership[config_id] += 1
                fractional[config_id] += 1.0 / len(oracle_set)
            if len(oracle_set) == 1:
                unique[oracle_set[0]] += 1

        for config_id in sorted(membership):
            winning_rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "n_units": n,
                    "config_id": config_id,
                    "oracle_membership_count": membership[config_id],
                    "oracle_fractional_count": fractional[config_id],
                    "unique_winner_count": unique[config_id],
                    "oracle_membership_rate": membership[config_id] / n if n else None,
                    "oracle_fractional_rate": fractional[config_id] / n if n else None,
                    "unique_winner_rate": unique[config_id] / n if n else None,
                }
            )

        for oracle_set, count in sorted(tie_sets.items()):
            tie_set_rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "n_units": n,
                    "oracle_config_set": oracle_set,
                    "count": count,
                    "rate": count / n if n else None,
                }
            )

        top_ties = sum(1 for unit in group if len(unit["oracle_set"]) > 1)
        margin_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n_units": n,
                "top_tie_rate": top_ties / n if n else None,
                "unique_winner_rate": 1.0 - (top_ties / n) if n else None,
                "top_2_margin_mean": mean(margins),
                "top_2_margin_median": median(margins),
                "top_2_margin_std": std(margins),
                "top_2_margin_min": min(margins) if margins else None,
                "top_2_margin_p10": percentile(margins, 0.10),
                "top_2_margin_p25": percentile(margins, 0.25),
                "top_2_margin_p75": percentile(margins, 0.75),
                "top_2_margin_p90": percentile(margins, 0.90),
                "top_2_margin_max": max(margins) if margins else None,
                "top_2_margin_values": json.dumps(margins),
            }
        )

    return winning_rows, tie_set_rows, margin_rows


def baseline_detail_rows(
    units: Sequence[Dict[str, Any]],
    global_baseline: Optional[Dict[str, Any]],
    objective_baselines: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    global_config = global_baseline["selected_config_id"] if global_baseline else None
    for unit in units:
        objective_baseline = objective_baselines.get(unit["learning_type"]) or global_baseline
        objective_config = objective_baseline["selected_config_id"] if objective_baseline else None
        global_utility = unit["utilities"].get(global_config) if global_config else None
        objective_utility = unit["utilities"].get(objective_config) if objective_config else None
        rows.append(
            {
                "episode_id": unit["episode_id"],
                "seed": unit["seed"],
                "meta_split": unit["meta_split"],
                "learning_type": unit["learning_type"],
                "oracle_config_set": config_set_id(unit["oracle_set"]),
                "oracle_utility": unit["oracle_utility"],
                "top_2_margin": unit["top_2_margin"],
                "global_baseline_config_id": global_config,
                "global_baseline_utility": global_utility,
                "global_to_oracle_regret": unit["oracle_utility"] - global_utility if global_utility is not None else None,
                "global_baseline_oracle_optimal": int(global_config in unit["oracle_set"]) if global_config else None,
                "objective_conditioned_config_id": objective_config,
                "objective_conditioned_utility": objective_utility,
                "objective_conditioned_to_oracle_regret": (
                    unit["oracle_utility"] - objective_utility if objective_utility is not None else None
                ),
                "objective_conditioned_oracle_optimal": (
                    int(objective_config in unit["oracle_set"]) if objective_config else None
                ),
            }
        )
    return rows


def summarize_baselines(
    detail_rows: Sequence[Dict[str, Any]],
    global_baseline: Optional[Dict[str, Any]],
    objective_baselines: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group_type, group_value, group_units_as_dicts in group_units(detail_rows):
        group = list(group_units_as_dicts)
        if not group:
            continue
        if group_type == "learning_type":
            objective_baseline = objective_baselines.get(group_value) or global_baseline
            objective_config_ids = objective_baseline["selected_config_id"] if objective_baseline else None
        else:
            objective_config_ids = json.dumps(
                {
                    key: value["selected_config_id"]
                    for key, value in sorted(objective_baselines.items())
                    if value is not None
                },
                sort_keys=True,
            )

        out.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n_units": len(group),
                "global_baseline_config_id": global_baseline["selected_config_id"] if global_baseline else None,
                "global_baseline_train_mean_utility": (
                    global_baseline["train_mean_utility"] if global_baseline else None
                ),
                "global_baseline_train_tie_set": (
                    config_set_id(global_baseline["tie_set"]) if global_baseline else None
                ),
                "objective_conditioned_config_id_or_map": objective_config_ids,
                "oracle_mean_observed_utility": mean(float(row["oracle_utility"]) for row in group),
                "global_baseline_mean_observed_utility": mean(
                    maybe_float(row["global_baseline_utility"]) for row in group
                ),
                "objective_conditioned_mean_observed_utility": mean(
                    maybe_float(row["objective_conditioned_utility"]) for row in group
                ),
                "global_to_oracle_regret_mean": mean(
                    maybe_float(row["global_to_oracle_regret"]) for row in group
                ),
                "objective_conditioned_to_oracle_regret_mean": mean(
                    maybe_float(row["objective_conditioned_to_oracle_regret"]) for row in group
                ),
                "global_oracle_optimal_fraction": mean(
                    maybe_float(row["global_baseline_oracle_optimal"]) for row in group
                ),
                "objective_conditioned_oracle_optimal_fraction": mean(
                    maybe_float(row["objective_conditioned_oracle_optimal"]) for row in group
                ),
            }
        )
    return out


def episode_utility_rows(units: Sequence[Dict[str, Any]], config_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    utility_cols = [f"utility__{config_id}" for config_id in config_ids]
    fieldnames = [
        "episode_id",
        "seed",
        "meta_split",
        "learning_type",
        "oracle_config_set",
        "oracle_utility",
        "top_2_margin",
        *utility_cols,
    ]
    rows: List[Dict[str, Any]] = []
    for unit in units:
        row = {
            "episode_id": unit["episode_id"],
            "seed": unit["seed"],
            "meta_split": unit["meta_split"],
            "learning_type": unit["learning_type"],
            "oracle_config_set": config_set_id(unit["oracle_set"]),
            "oracle_utility": unit["oracle_utility"],
            "top_2_margin": unit["top_2_margin"],
        }
        for config_id in config_ids:
            row[f"utility__{config_id}"] = unit["utilities"].get(config_id)
        rows.append(row)
    return rows, fieldnames


def parse_eval_splits(values: Sequence[str]) -> Optional[set[str]]:
    if any(value.lower() == "all" for value in values):
        return None
    return {str(value) for value in values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze adaptation-compiler headroom with tie-aware oracles.")
    parser.add_argument("--records", default="artifacts/llama/geometry/geometry_records_by_seed.csv")
    parser.add_argument("--config_ids", nargs="+", default=None)
    parser.add_argument("--output_dir", default="outputs/release_verification/llama/headroom_analysis")
    parser.add_argument("--eval_splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--utility_columns", nargs="+", default=UTILITY_COLUMNS)
    parser.add_argument("--tie_tolerance", type=float, default=1e-12)
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.records)
    units, warnings = build_episode_units(
        records,
        config_ids=args.config_ids,
        utility_columns=args.utility_columns,
        tie_tolerance=args.tie_tolerance,
        allow_incomplete=args.allow_incomplete,
    )
    config_ids = list(args.config_ids or sorted({cfg for unit in units for cfg in unit["utilities"]}))
    train_units = [unit for unit in units if unit["meta_split"] == "train"]
    eval_splits = parse_eval_splits(args.eval_splits)
    eval_units = [unit for unit in units if eval_splits is None or unit["meta_split"] in eval_splits]
    if not train_units:
        warnings.append("No TRAIN units were available; fixed baselines could not be learned.")
    if not eval_units:
        warnings.append("No evaluation units matched --eval_splits.")

    global_baseline = learn_fixed_baseline(train_units, tie_tolerance=args.tie_tolerance)
    objective_baselines: Dict[str, Optional[Dict[str, Any]]] = {}
    for learning_type in sorted({unit["learning_type"] for unit in units}):
        objective_train = [unit for unit in train_units if unit["learning_type"] == learning_type]
        objective_baselines[learning_type] = learn_fixed_baseline(
            objective_train,
            tie_tolerance=args.tie_tolerance,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows, episode_fields = episode_utility_rows(eval_units, config_ids)
    winning_rows, tie_set_rows, margin_rows = summarize_winners(eval_units)
    baseline_detail = baseline_detail_rows(eval_units, global_baseline, objective_baselines)
    baseline_summary = summarize_baselines(baseline_detail, global_baseline, objective_baselines)

    write_csv(output_dir / "headroom_episode_utilities.csv", episode_rows, episode_fields)
    write_csv(
        output_dir / "headroom_winning_config_counts.csv",
        winning_rows,
        [
            "group_type",
            "group_value",
            "n_units",
            "config_id",
            "oracle_membership_count",
            "oracle_fractional_count",
            "unique_winner_count",
            "oracle_membership_rate",
            "oracle_fractional_rate",
            "unique_winner_rate",
        ],
    )
    write_csv(
        output_dir / "headroom_tie_sets.csv",
        tie_set_rows,
        ["group_type", "group_value", "n_units", "oracle_config_set", "count", "rate"],
    )
    write_csv(
        output_dir / "headroom_top2_margins.csv",
        margin_rows,
        [
            "group_type",
            "group_value",
            "n_units",
            "top_tie_rate",
            "unique_winner_rate",
            "top_2_margin_mean",
            "top_2_margin_median",
            "top_2_margin_std",
            "top_2_margin_min",
            "top_2_margin_p10",
            "top_2_margin_p25",
            "top_2_margin_p75",
            "top_2_margin_p90",
            "top_2_margin_max",
            "top_2_margin_values",
        ],
    )
    write_csv(
        output_dir / "headroom_baseline_detail.csv",
        baseline_detail,
        [
            "episode_id",
            "seed",
            "meta_split",
            "learning_type",
            "oracle_config_set",
            "oracle_utility",
            "top_2_margin",
            "global_baseline_config_id",
            "global_baseline_utility",
            "global_to_oracle_regret",
            "global_baseline_oracle_optimal",
            "objective_conditioned_config_id",
            "objective_conditioned_utility",
            "objective_conditioned_to_oracle_regret",
            "objective_conditioned_oracle_optimal",
        ],
    )
    write_csv(
        output_dir / "headroom_baseline_summary.csv",
        baseline_summary,
        [
            "group_type",
            "group_value",
            "n_units",
            "global_baseline_config_id",
            "global_baseline_train_mean_utility",
            "global_baseline_train_tie_set",
            "objective_conditioned_config_id_or_map",
            "oracle_mean_observed_utility",
            "global_baseline_mean_observed_utility",
            "objective_conditioned_mean_observed_utility",
            "global_to_oracle_regret_mean",
            "objective_conditioned_to_oracle_regret_mean",
            "global_oracle_optimal_fraction",
            "objective_conditioned_oracle_optimal_fraction",
        ],
    )

    summary = {
        "records": str(args.records),
        "config_ids": config_ids,
        "utility": f"mean({', '.join(args.utility_columns)})",
        "eval_splits": sorted(eval_splits) if eval_splits is not None else "all",
        "n_records": len(records),
        "n_units": len(units),
        "n_train_units": len(train_units),
        "n_eval_units": len(eval_units),
        "global_baseline": global_baseline,
        "objective_baselines": objective_baselines,
        "warnings": warnings,
        "outputs": {
            "episode_utilities": str(output_dir / "headroom_episode_utilities.csv"),
            "winning_config_counts": str(output_dir / "headroom_winning_config_counts.csv"),
            "tie_sets": str(output_dir / "headroom_tie_sets.csv"),
            "top2_margins": str(output_dir / "headroom_top2_margins.csv"),
            "baseline_detail": str(output_dir / "headroom_baseline_detail.csv"),
            "baseline_summary": str(output_dir / "headroom_baseline_summary.csv"),
        },
    }
    (output_dir / "headroom_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for warning in warnings:
        print(f"[warn] {warning}")
    print(f"Wrote headroom analysis to {output_dir}")


if __name__ == "__main__":
    main()
