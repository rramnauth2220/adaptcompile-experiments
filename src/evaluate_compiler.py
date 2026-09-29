#!/usr/bin/env python3
"""Evaluate compiler selection regret against observed oracle outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    from .compiler_common import read_jsonl, write_csv
except ImportError:  # pragma: no cover
    from compiler_common import read_jsonl, write_csv


DEFAULT_WEIGHTS = {"acquisition": 1.0, "transfer": 1.0, "boundedness": 1.0, "preservation": 1.0}


def read_table(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_utility_weights(values: Sequence[str] | None) -> Dict[str, float]:
    if not values:
        return dict(DEFAULT_WEIGHTS)
    out: Dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Utility weights must look like metric=weight, got {item!r}")
        metric, weight = item.split("=", 1)
        out[metric] = float(weight)
    return out


def utility(row: Dict[str, Any], weights: Dict[str, float], prefix: str = "") -> float:
    total = 0.0
    weight_sum = 0.0
    for metric, weight in weights.items():
        if float(weight) == 0.0:
            continue
        key = f"{prefix}{metric}"
        value = row.get(key)
        if value in (None, ""):
            raise ValueError(
                f"Missing required utility metric {key!r} for config {row.get('config_id')!r}. "
                "Pass explicit --utility_weights that exclude unavailable historical metrics."
            )
        total += float(value) * weight
        weight_sum += abs(weight)
    return total / weight_sum if weight_sum else 0.0


def cost(row: Dict[str, Any]) -> float:
    for key in ["parameter_cost", "approximate_parameter_cost", "trainable_parameters"]:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return 0.0


def feasible_rows(
    rows: Sequence[Dict[str, Any]],
    max_parameter_cost: float | None = None,
) -> List[Dict[str, Any]]:
    candidates = [
        row for row in rows
        if max_parameter_cost is None or cost(row) <= max_parameter_cost
    ]
    if not candidates:
        raise ValueError("No candidate configurations satisfy the parameter-cost constraint.")
    return candidates


def choose_best(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    prefix: str,
    max_parameter_cost: float | None = None,
) -> Dict[str, Any]:
    """Choose one executable program.

    Compiler selection remains single-valued. Ties are resolved deterministically
    by config_id so execution always returns one configuration. Observed oracle
    evaluation is set-valued and handled by oracle_config_set().
    """
    candidates = feasible_rows(rows, max_parameter_cost=max_parameter_cost)
    return max(
        candidates,
        key=lambda row: (utility(row, weights, prefix=prefix), str(row["config_id"])),
    )


def oracle_config_set(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    prefix: str = "",
    max_parameter_cost: float | None = None,
    tie_tolerance: float = 1e-12,
) -> Tuple[set[str], float]:
    """Return all configurations tied for best utility within tolerance."""
    candidates = feasible_rows(rows, max_parameter_cost=max_parameter_cost)
    scored = [
        (str(row["config_id"]), utility(row, weights, prefix=prefix))
        for row in candidates
    ]
    best_utility = max(value for _, value in scored)
    oracle_ids = {
        config_id
        for config_id, value in scored
        if best_utility - value <= tie_tolerance
    }
    return oracle_ids, best_utility


def top_k_config_ids(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    prefix: str,
    k: int,
    tie_tolerance: float = 1e-12,
    max_parameter_cost: float | None = None,
) -> set[str]:
    """Return top-k configs, including all ties at the kth utility boundary."""
    if k <= 0:
        return set()
    candidates = feasible_rows(rows, max_parameter_cost=max_parameter_cost)
    scored = sorted(
        [
            (str(row["config_id"]), utility(row, weights, prefix=prefix))
            for row in candidates
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    boundary_index = min(k, len(scored)) - 1
    boundary_utility = scored[boundary_index][1]
    return {
        config_id
        for config_id, value in scored
        if value >= boundary_utility - tie_tolerance
    }

def globally_best_config(train_rows: Sequence[Dict[str, Any]], weights: Dict[str, float]) -> str | None:
    by_config: Dict[str, List[float]] = defaultdict(list)
    for row in train_rows:
        by_config[str(row["config_id"])].append(utility(row, weights, prefix=""))
    if not by_config:
        return None
    return max(by_config, key=lambda cfg: sum(by_config[cfg]) / len(by_config[cfg]))


def evaluate_predictions(
    rows: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    max_parameter_cost: float | None = None,
    top_k: int = 2,
    seed: int = 2026,
    train_rows: Sequence[Dict[str, Any]] | None = None,
    tie_tolerance: float = 1e-12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["episode_id"]), int(row.get("seed", 0)))].append(row)

    global_best = globally_best_config(train_rows or [], weights)
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for (episode_id, seed_value), group_rows in sorted(grouped.items()):
        compiler = choose_best(
            group_rows,
            weights,
            prefix="predicted_",
            max_parameter_cost=max_parameter_cost,
        )
        oracle_ids, oracle_obs = oracle_config_set(
            group_rows,
            weights,
            prefix="",
            max_parameter_cost=max_parameter_cost,
            tie_tolerance=tie_tolerance,
        )
        compiler_obs = utility(compiler, weights, prefix="")
        oracle_topk = top_k_config_ids(
            group_rows,
            weights,
            prefix="",
            k=top_k,
            tie_tolerance=tie_tolerance,
            max_parameter_cost=max_parameter_cost,
        )

        full_rows = [row for row in group_rows if str(row["config_id"]).startswith("full")]
        full = full_rows[0] if full_rows else None
        global_row = next((row for row in group_rows if row["config_id"] == global_best), None)
        random_row = rng.choice(list(group_rows))

        result = {
            "episode_id": episode_id,
            "seed": seed_value,
            "selected_config_id": compiler["config_id"],
            # Backward-compatible representative. The scientifically meaningful
            # oracle is the full set in oracle_config_ids.
            "oracle_config_id": sorted(oracle_ids)[0],
            "oracle_config_ids": "|".join(sorted(oracle_ids)),
            "oracle_config_set": "|".join(sorted(oracle_ids)),
            "oracle_set_size": len(oracle_ids),
            "selected_observed_utility": compiler_obs,
            "oracle_observed_utility": oracle_obs,
            "oracle_regret": oracle_obs - compiler_obs,
            # Preserve the historical column name, but make recovery tie-aware.
            "exact_top1_recovery": str(compiler["config_id"]) in oracle_ids,
            "oracle_recovery": str(compiler["config_id"]) in oracle_ids,
            "top_k_recovery": str(compiler["config_id"]) in oracle_topk,
            "top_k": top_k,
            "top_k_oracle_config_ids": "|".join(sorted(oracle_topk)),
            "selected_predicted_utility": utility(compiler, weights, prefix="predicted_"),
            "oracle_predicted_utility": max(
                utility(row, weights, prefix="predicted_")
                for row in group_rows
                if str(row["config_id"]) in oracle_ids
            ),
            "full_stack_observed_utility": utility(full, weights, prefix="") if full else None,
            "globally_best_training_config_id": global_best,
            "globally_best_training_observed_utility": utility(global_row, weights, prefix="") if global_row else None,
            "random_config_id": random_row["config_id"],
            "random_observed_utility": utility(random_row, weights, prefix=""),
        }
        for metric in weights:
            result[f"selected_{metric}"] = compiler.get(metric)
            oracle_values = [
                row.get(metric)
                for row in group_rows
                if str(row["config_id"]) in oracle_ids and row.get(metric) not in (None, "")
            ]
            result[f"oracle_{metric}"] = (
                sum(float(v) for v in oracle_values) / len(oracle_values)
                if oracle_values
                else None
            )
        out.append(result)

    n = len(out)
    summary = {
        "n_episode_seed_groups": n,
        "mean_selected_observed_utility": sum(float(r["selected_observed_utility"]) for r in out) / n if n else None,
        "mean_oracle_observed_utility": sum(float(r["oracle_observed_utility"]) for r in out) / n if n else None,
        "mean_oracle_regret": sum(float(r["oracle_regret"]) for r in out) / n if n else None,
        "exact_top1_recovery_rate": sum(bool(r["exact_top1_recovery"]) for r in out) / n if n else None,
        "oracle_recovery_rate": sum(bool(r["oracle_recovery"]) for r in out) / n if n else None,
        "top_k_recovery_rate": sum(bool(r["top_k_recovery"]) for r in out) / n if n else None,
        "mean_oracle_set_size": sum(int(r["oracle_set_size"]) for r in out) / n if n else None,
        "utility_weights": weights,
        "max_parameter_cost": max_parameter_cost,
        "tie_tolerance": tie_tolerance,
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compiler selection regret.")
    parser.add_argument("--input", default="outputs/compiler/predictor/predictions_test.csv")
    parser.add_argument("--train_records", default=None)
    parser.add_argument("--output_csv", default="outputs/compiler/compiler_evaluation.csv")
    parser.add_argument("--output_json", default="outputs/compiler/compiler_evaluation_summary.json")
    parser.add_argument("--utility_weights", nargs="*", default=None)
    parser.add_argument("--max_parameter_cost", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-12,
        help="Observed-utility tolerance for treating configurations as tied.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.input)
    train_rows = read_table(args.train_records) if args.train_records else []
    weights = parse_utility_weights(args.utility_weights)
    detail, summary = evaluate_predictions(
        rows,
        weights=weights,
        max_parameter_cost=args.max_parameter_cost,
        top_k=args.top_k,
        seed=args.seed,
        train_rows=train_rows,
        tie_tolerance=args.tie_tolerance,
    )
    write_csv(args.output_csv, detail, list(detail[0].keys()) if detail else [])
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
