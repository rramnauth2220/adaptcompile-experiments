#!/usr/bin/env python3
"""Summaries for the Gemma adaptation-compiler replication.

The heavy lifting remains in existing compiler modules. This script provides
Gemma-specific file naming, calibration selection, pilot headroom summaries, and
a descriptive Llama-vs-Gemma comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
REPRODUCE_DIR = PROJECT_ROOT / "scripts" / "reproduce"
for path in [SRC_DIR, REPRODUCE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_compiler_headroom import (  # noqa: E402
    baseline_detail_rows,
    build_episode_units,
    learn_fixed_baseline,
    mean,
    median,
    summarize_baselines,
    summarize_winners,
)
from build_geometry_records import FIELDS as RAW_GEOMETRY_FIELDS  # noqa: E402
from build_geometry_records import build_records  # noqa: E402
from causal_content_scoring import CAUSAL_SCORE_VERSION  # noqa: E402
from compiler_backbones import PRIMARY_COMPILER_CONFIG_IDS  # noqa: E402
from compiler_common import read_csv, write_csv, write_jsonl  # noqa: E402
from prepare_geometry_dataset import (  # noqa: E402
    FIELDS as GEOMETRY_DATASET_FIELDS,
    expected_seed_map,
    prepare_geometry_dataset,
)


OUTCOMES = ["acquisition", "transfer", "boundedness", "preservation"]


def read_table(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    return read_csv(path)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def utility(row: Mapping[str, Any]) -> float | None:
    vals = [maybe_float(row.get(metric)) for metric in OUTCOMES]
    if any(value is None for value in vals):
        return None
    return sum(float(value) for value in vals) / len(vals)


def parse_grad_accum_from_path(path: str | Path) -> int | None:
    match = re.search(r"ga[_-](\d+)", str(path).replace("\\", "/"))
    return int(match.group(1)) if match else None


def collect_records(input_glob: str, causal_metric_version: str | None = None) -> List[Dict[str, Any]]:
    records = build_records([input_glob], causal_metric_version=causal_metric_version)
    if not records:
        raise ValueError(f"No geometry records found for {input_glob}")
    return records


def collect_calibration_rows(results_root: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], str | None]:
    """Collect calibration rows without cross-GA preservation collisions.

    The same episode/config/seed is intentionally evaluated for every GA value,
    so preservation summaries must be joined one GA directory at a time.
    """

    rows: List[Dict[str, Any]] = []
    for ga_dir in sorted(results_root.glob("ga_*")):
        ga = parse_grad_accum_from_path(ga_dir)
        if ga is None:
            continue
        eval_paths = sorted((ga_dir / "runs").glob("**/evaluation.jsonl"))
        if not eval_paths:
            continue
        for record in build_records([str(path) for path in eval_paths]):
            out = dict(record)
            out["grad_accum"] = ga
            out["balanced_utility"] = utility(record)
            rows.append(out)
    if rows:
        return rows, None

    cached_runs = output_dir / "calibration_runs.csv"
    if not cached_runs.exists() and (results_root / "calibration_runs.csv").exists():
        cached_runs = results_root / "calibration_runs.csv"
    if cached_runs.exists():
        warning = (
            f"Raw ga_*/runs/evaluation.jsonl files were not found; using cached {cached_runs}. "
            "If this CSV was produced by an older summarizer, preservation_source_file may not "
            "identify the correct GA-specific preservation file."
        )
        return read_csv(cached_runs), warning

    raise ValueError(f"No calibration records found under {results_root}")


def write_raw_and_dataset(
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
    config_ids: Sequence[str],
    *,
    allow_incomplete_smoke_test: bool = False,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw_geometry.jsonl"
    raw_csv = output_dir / "raw_geometry.csv"
    dataset_csv = output_dir / "geometry_dataset.csv"
    diagnostics_json = output_dir / "geometry_dataset.diagnostics.json"

    write_jsonl(raw_jsonl, records)
    write_csv(raw_csv, list(records), RAW_GEOMETRY_FIELDS)
    dataset, diagnostics = prepare_geometry_dataset(
        records,
        config_ids=config_ids,
        expected_seeds_by_split=expected_seed_map(
            train=[11],
            validation=[11],
            test=[11, 22, 33],
        ),
        strict=not allow_incomplete_smoke_test,
    )
    write_csv(dataset_csv, dataset, GEOMETRY_DATASET_FIELDS)
    write_json(diagnostics_json, diagnostics)
    print(f"Wrote raw geometry: {raw_csv}")
    print(f"Wrote seed-aggregated geometry: {dataset_csv}")
    return raw_csv, dataset_csv


def command_prepare(args: argparse.Namespace) -> None:
    records = collect_records(args.input_glob, causal_metric_version=getattr(args, "causal_metric_version", None))
    write_raw_and_dataset(
        records,
        Path(args.output_dir),
        args.config_ids,
        allow_incomplete_smoke_test=args.allow_incomplete_smoke_test,
    )


def manifest_expected_counts(results_root: Path) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for path in sorted(results_root.glob("ga_*/runs/compiler_geometry_pilot_manifest.csv")):
        ga = parse_grad_accum_from_path(path)
        if ga is None:
            continue
        rows = read_csv(path)
        counts[ga] += len(rows)
    return counts


def command_calibration(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_warning = collect_calibration_rows(Path(args.results_root), output_dir)

    runs_csv = output_dir / "calibration_runs.csv"
    write_csv(runs_csv, rows, ["grad_accum", *RAW_GEOMETRY_FIELDS, "balanced_utility"])

    expected_counts = manifest_expected_counts(Path(args.results_root))
    summary_rows: List[Dict[str, Any]] = []
    for ga in sorted({int(row["grad_accum"]) for row in rows} | set(expected_counts)):
        group = [row for row in rows if int(row["grad_accum"]) == ga]
        observed_units = {
            (row["episode_id"], row["config_id"], int(row["seed"]))
            for row in group
        }
        expected = expected_counts.get(ga, len(observed_units))
        summary = {
            "grad_accum": ga,
            "acquisition": mean(maybe_float(row.get("acquisition")) for row in group),
            "transfer": mean(maybe_float(row.get("transfer")) for row in group),
            "boundedness": mean(maybe_float(row.get("boundedness")) for row in group),
            "preservation": mean(maybe_float(row.get("preservation")) for row in group),
            "balanced_utility": mean(maybe_float(row.get("balanced_utility")) for row in group),
            "number_of_episodes": len({row["episode_id"] for row in group}),
            "observed_episode_program_seed_units": len(observed_units),
            "expected_episode_program_seed_units": expected,
            "failures": max(0, expected - len(observed_units)),
        }
        summary_rows.append(summary)

    summary_csv = output_dir / "calibration_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "grad_accum",
            "acquisition",
            "transfer",
            "boundedness",
            "preservation",
            "balanced_utility",
            "number_of_episodes",
            "observed_episode_program_seed_units",
            "expected_episode_program_seed_units",
            "failures",
        ],
    )
    by_objective_rows = summarize_calibration_by_objective(rows)
    by_objective_csv = output_dir / "calibration_by_objective.csv"
    write_csv(
        by_objective_csv,
        by_objective_rows,
        [
            "grad_accum",
            "objective",
            "acquisition",
            "transfer",
            "boundedness",
            "preservation",
            "balanced_utility",
            "number_of_episodes",
        ],
    )

    selected = select_calibration_schedule(summary_rows, args)
    if source_warning:
        selected["source_warning"] = source_warning
    selected_path = output_dir / "selected_schedule.json"
    write_json(selected_path, selected)
    print(f"Wrote {runs_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {by_objective_csv}")
    print(f"Wrote {selected_path}")
    if source_warning:
        print(f"[warn] {source_warning}")
    if selected["status"] != "selected":
        print(f"[warn] Calibration inconclusive: {selected['reason']}")


def select_calibration_schedule(summary_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    complete = [
        row for row in summary_rows
        if int(row.get("failures") or 0) <= int(args.max_failures)
        and maybe_float(row.get("acquisition")) is not None
        and maybe_float(row.get("transfer")) is not None
        and maybe_float(row.get("boundedness")) is not None
        and maybe_float(row.get("preservation")) is not None
    ]
    if not complete:
        return {
            "status": "inconclusive",
            "reason": "No complete calibration condition with finite A/T/B/P metrics.",
            "selection_rule": calibration_rule_text(args),
            "manual_review": calibration_manual_review(summary_rows),
            "candidate_summaries": list(summary_rows),
        }

    best_acq = max(float(row["acquisition"]) for row in complete)
    best_transfer = max(float(row["transfer"]) for row in complete)
    feasible = [
        row for row in complete
        if float(row["acquisition"]) >= max(args.min_acquisition, best_acq - args.max_acquisition_drop)
        and float(row["transfer"]) >= max(args.min_transfer, best_transfer - args.max_transfer_drop)
        and float(row["boundedness"]) >= args.min_boundedness
        and float(row["preservation"]) >= args.min_preservation
    ]
    if not feasible:
        return {
            "status": "inconclusive",
            "reason": "No calibration condition satisfied the acquisition/transfer/boundedness/preservation gates.",
            "selection_rule": calibration_rule_text(args),
            "manual_review": calibration_manual_review(summary_rows),
            "candidate_summaries": list(summary_rows),
        }

    # Larger gradient accumulation is less update-intensive for a fixed epoch
    # count and batch size, so choose the largest feasible value.
    chosen = max(feasible, key=lambda row: (int(row["grad_accum"]), float(row.get("balanced_utility") or -1.0)))
    return {
        "status": "selected",
        "gradient_accumulation_steps": int(chosen["grad_accum"]),
        "selected_summary": chosen,
        "selection_rule": calibration_rule_text(args),
        "candidate_summaries": list(summary_rows),
    }


def summarize_calibration_by_objective(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["grad_accum"]), str(row["learning_type"]))].append(row)
    out = []
    for (ga, objective), group in sorted(grouped.items()):
        out.append(
            {
                "grad_accum": ga,
                "objective": objective,
                "acquisition": mean(maybe_float(row.get("acquisition")) for row in group),
                "transfer": mean(maybe_float(row.get("transfer")) for row in group),
                "boundedness": mean(maybe_float(row.get("boundedness")) for row in group),
                "preservation": mean(maybe_float(row.get("preservation")) for row in group),
                "balanced_utility": mean(maybe_float(row.get("balanced_utility")) for row in group),
                "number_of_episodes": len({row["episode_id"] for row in group}),
            }
        )
    return out


def calibration_manual_review(summary_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    complete = [
        row for row in summary_rows
        if maybe_float(row.get("balanced_utility")) is not None
        and int(row.get("failures") or 0) == 0
    ]
    if not complete:
        return {"available": False}
    best_balanced = max(complete, key=lambda row: float(row["balanced_utility"]))
    best_acq_transfer = max(
        complete,
        key=lambda row: (
            (float(row.get("acquisition") or 0.0) + float(row.get("transfer") or 0.0)) / 2.0,
            float(row.get("balanced_utility") or 0.0),
        ),
    )
    least_update_with_transfer = max(
        complete,
        key=lambda row: (
            float(row.get("transfer") or 0.0) >= 0.5,
            int(row.get("grad_accum") or 0),
            float(row.get("balanced_utility") or 0.0),
        ),
    )
    return {
        "available": True,
        "best_balanced_utility_grad_accum": int(best_balanced["grad_accum"]),
        "best_balanced_utility": float(best_balanced["balanced_utility"]),
        "best_acquisition_transfer_grad_accum": int(best_acq_transfer["grad_accum"]),
        "best_acquisition_transfer_mean": (
            float(best_acq_transfer.get("acquisition") or 0.0)
            + float(best_acq_transfer.get("transfer") or 0.0)
        )
        / 2.0,
        "least_update_candidate_with_transfer_at_least_0_5": int(least_update_with_transfer["grad_accum"]),
        "note": (
            "These are diagnostics for human review only. status remains inconclusive "
            "unless a condition satisfies the configured calibration gates."
        ),
    }


def calibration_rule_text(args: argparse.Namespace) -> str:
    return (
        "Choose the largest gradient_accumulation_steps value with no more than "
        f"{args.max_failures} failures, acquisition >= max({args.min_acquisition}, best - {args.max_acquisition_drop}), "
        f"transfer >= max({args.min_transfer}, best - {args.max_transfer_drop}), "
        f"boundedness >= {args.min_boundedness}, and preservation >= {args.min_preservation}."
    )


def command_pilot(args: argparse.Namespace) -> None:
    records = collect_records(args.input_glob, causal_metric_version=getattr(args, "causal_metric_version", None))
    raw_csv, dataset_csv = write_raw_and_dataset(
        records,
        Path(args.output_dir),
        args.config_ids,
        allow_incomplete_smoke_test=False,
    )
    rows = read_csv(dataset_csv)
    units, warnings = build_episode_units(
        rows,
        config_ids=args.config_ids,
        utility_columns=OUTCOMES,
        tie_tolerance=args.tie_tolerance,
        allow_incomplete=False,
    )
    train_units = [unit for unit in units if unit["meta_split"] == "train"]
    eval_units = list(units)
    global_baseline = learn_fixed_baseline(train_units, tie_tolerance=args.tie_tolerance)
    objective_baselines = {
        learning_type: learn_fixed_baseline(
            [unit for unit in train_units if unit["learning_type"] == learning_type],
            tie_tolerance=args.tie_tolerance,
        )
        for learning_type in sorted({unit["learning_type"] for unit in units})
    }
    baseline_detail = baseline_detail_rows(eval_units, global_baseline, objective_baselines)
    baseline_summary = summarize_baselines(baseline_detail, global_baseline, objective_baselines)
    winning_rows, tie_set_rows, margin_rows = summarize_winners(eval_units)

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "oracle_winners.csv",
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
    by_objective = [
        {
            "objective": row["group_value"],
            "oracle_utility": row["oracle_mean_observed_utility"],
            "global_utility": row["global_baseline_mean_observed_utility"],
            "objective_fixed_utility": row["objective_conditioned_mean_observed_utility"],
            "global_regret": row["global_to_oracle_regret_mean"],
            "objective_fixed_regret": row["objective_conditioned_to_oracle_regret_mean"],
            "global_oracle_optimal_rate": row["global_oracle_optimal_fraction"],
            "objective_oracle_optimal_rate": row["objective_conditioned_oracle_optimal_fraction"],
        }
        for row in baseline_summary
        if row["group_type"] == "learning_type"
    ]
    write_csv(
        output_dir / "selection_by_objective.csv",
        by_objective,
        [
            "objective",
            "oracle_utility",
            "global_utility",
            "objective_fixed_utility",
            "global_regret",
            "objective_fixed_regret",
            "global_oracle_optimal_rate",
            "objective_oracle_optimal_rate",
        ],
    )

    spreads = [max(unit["utilities"].values()) - min(unit["utilities"].values()) for unit in eval_units]
    overall = next((row for row in baseline_summary if row["group_type"] == "overall"), None)
    summary = {
        "raw_geometry_csv": str(raw_csv),
        "geometry_dataset_csv": str(dataset_csv),
        "n_episode_units": len(eval_units),
        "n_train_units_used_for_fixed_policies": len(train_units),
        "mean_max_minus_min_program_utility": mean(spreads),
        "median_max_minus_min_program_utility": median(spreads),
        "effectively_tied_episode_fraction": mean(1.0 if len(unit["oracle_set"]) > 1 else 0.0 for unit in eval_units),
        "overall_selection": overall,
        "oracle_winner_shares": [
            row for row in winning_rows if row["group_type"] == "overall"
        ],
        "tie_sets": tie_set_rows,
        "margin_summary": margin_rows,
        "warnings": warnings,
    }
    write_json(output_dir / "selection_summary.json", summary)
    print(f"Wrote {output_dir / 'selection_summary.json'}")
    print(f"Wrote {output_dir / 'selection_by_objective.csv'}")
    print(f"Wrote {output_dir / 'oracle_winners.csv'}")


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_metric(rows: List[Dict[str, Any]], backbone: str, scope: str, metric: str, value: Any, source: Path) -> None:
    if value in (None, ""):
        return
    rows.append(
        {
            "backbone": backbone,
            "scope": scope,
            "metric": metric,
            "value": value,
            "source_file": str(source),
        }
    )


def collect_backbone_comparison_metrics(backbone: str, root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    headroom = root / "geometry" / "headroom"
    if not headroom.exists():
        headroom = root / "experiment1" / "full_sweep_headroom"
    if not headroom.exists():
        headroom = root / "headroom_analysis"
    baseline_summary_path = headroom / "headroom_baseline_summary.csv"
    for row in read_table(baseline_summary_path):
        if row.get("group_type") == "overall":
            for src_key, out_key in [
                ("oracle_mean_observed_utility", "oracle_utility"),
                ("global_baseline_mean_observed_utility", "global_utility"),
                ("objective_conditioned_mean_observed_utility", "objective_fixed_utility"),
                ("global_to_oracle_regret_mean", "global_regret"),
                ("objective_conditioned_to_oracle_regret_mean", "objective_fixed_regret"),
                ("global_oracle_optimal_fraction", "global_oracle_optimal_rate"),
                ("objective_conditioned_oracle_optimal_fraction", "objective_oracle_optimal_rate"),
            ]:
                add_metric(rows, backbone, "selection_problem", out_key, row.get(src_key), baseline_summary_path)

    margins_path = headroom / "headroom_top2_margins.csv"
    for row in read_table(margins_path):
        if row.get("group_type") == "overall":
            add_metric(rows, backbone, "selection_problem", "tie_frequency", row.get("top_tie_rate"), margins_path)
            add_metric(rows, backbone, "selection_problem", "mean_oracle_margin", row.get("top_2_margin_mean"), margins_path)

    exp2 = root / "prediction"
    if not exp2.exists():
        exp2 = root / "experiment2" / "full_auto"
    if not exp2.exists():
        exp2 = root / "full" / "experiment2" / "full_auto"
    pred = read_json_file(exp2 / "prediction_metrics.json")
    ranking = read_json_file(exp2 / "ranking_metrics.json")
    primary_pred = pred.get("primary", pred)
    primary_rank = ranking.get("primary", ranking)
    for src_key, out_key in [
        ("overall_mae", "geometry_mae"),
        ("utility_pearson_correlation", "utility_correlation"),
        ("mean_episode_spearman", "spearman"),
        ("pairwise_ranking_accuracy", "pairwise_accuracy"),
        ("top1_oracle_recovery", "top1"),
        ("top2_oracle_recovery", "top2"),
    ]:
        add_metric(rows, backbone, "prediction", out_key, primary_pred.get(src_key, primary_rank.get(src_key)), exp2)

    exp3 = root / "selection"
    if not exp3.exists():
        exp3 = root / "experiment3"
    if not exp3.exists():
        exp3 = root / "full" / "experiment3"
    compiler = read_json_file(exp3 / "compiler_evaluation_summary.json")
    for src_key, out_key in [
        ("mean_selected_observed_utility", "compiler_utility"),
        ("mean_oracle_regret", "compiler_regret"),
        ("oracle_recovery_rate", "compiler_top1"),
        ("top_k_recovery_rate", "compiler_top2"),
    ]:
        add_metric(rows, backbone, "compiler", out_key, compiler.get(src_key), exp3)
    return rows


def command_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        *collect_backbone_comparison_metrics("llama", Path(args.llama_root)),
        *collect_backbone_comparison_metrics("gemma", Path(args.gemma_root)),
    ]
    output_csv = output_dir / "llama_vs_gemma_summary.csv"
    write_csv(output_csv, rows, ["backbone", "scope", "metric", "value", "source_file"])

    by_objective_rows = compare_matched_geometry(Path(args.llama_root), Path(args.gemma_root))
    write_csv(
        output_dir / "llama_vs_gemma_by_objective.csv",
        by_objective_rows,
        [
            "objective",
            "matched_episode_config_pairs",
            "utility_correlation",
            "same_oracle_episode_fraction",
            "mean_llama_oracle_margin",
            "mean_gemma_oracle_margin",
        ],
    )
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_dir / 'llama_vs_gemma_by_objective.csv'}")


def geometry_dataset_path(root: Path) -> Path | None:
    candidates = [
        root / "geometry" / "geometry_dataset.csv",
        root / "experiment2" / "geometry_dataset.csv",
        root / "experiment2" / "full_geometry_dataset.csv",
        root / "full" / "geometry_dataset.csv",
        root / "geometry_dataset.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def compare_matched_geometry(llama_root: Path, gemma_root: Path) -> List[Dict[str, Any]]:
    llama_path = geometry_dataset_path(llama_root)
    gemma_path = geometry_dataset_path(gemma_root)
    if llama_path is None or gemma_path is None:
        return []
    llama_rows = read_csv(llama_path)
    gemma_rows = read_csv(gemma_path)
    llama_by_key = {
        (row["episode_id"], row["config_id"]): row
        for row in llama_rows
    }
    gemma_by_key = {
        (row["episode_id"], row["config_id"]): row
        for row in gemma_rows
    }
    objectives = sorted({row["learning_type"] for row in llama_rows} | {row["learning_type"] for row in gemma_rows})
    out = []
    for objective in objectives:
        keys = [
            key for key in sorted(set(llama_by_key) & set(gemma_by_key))
            if llama_by_key[key].get("learning_type") == objective or gemma_by_key[key].get("learning_type") == objective
        ]
        xs = [maybe_float(llama_by_key[key].get("utility")) for key in keys]
        ys = [maybe_float(gemma_by_key[key].get("utility")) for key in keys]
        paired = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        llama_oracles = oracle_by_episode([row for row in llama_rows if row.get("learning_type") == objective])
        gemma_oracles = oracle_by_episode([row for row in gemma_rows if row.get("learning_type") == objective])
        shared_episodes = sorted(set(llama_oracles) & set(gemma_oracles))
        same_oracle = [
            1.0 if llama_oracles[episode_id]["oracle_set"] == gemma_oracles[episode_id]["oracle_set"] else 0.0
            for episode_id in shared_episodes
        ]
        out.append(
            {
                "objective": objective,
                "matched_episode_config_pairs": len(paired),
                "utility_correlation": pearson([x for x, _ in paired], [y for _, y in paired]),
                "same_oracle_episode_fraction": mean(same_oracle),
                "mean_llama_oracle_margin": mean(row["margin"] for row in llama_oracles.values()),
                "mean_gemma_oracle_margin": mean(row["margin"] for row in gemma_oracles.values()),
            }
        )
    return out


def oracle_by_episode(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["episode_id"]].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for episode_id, group in buckets.items():
        scored = [(row["config_id"], maybe_float(row.get("utility"))) for row in group]
        scored = [(cfg, value) for cfg, value in scored if value is not None]
        if not scored:
            continue
        best = max(value for _, value in scored)
        ordered = sorted([value for _, value in scored], reverse=True)
        out[episode_id] = {
            "oracle_set": "|".join(sorted(cfg for cfg, value in scored if best - value <= 1e-12)),
            "margin": ordered[0] - ordered[1] if len(ordered) >= 2 else 0.0,
        }
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Gemma compiler replication artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build raw and seed-aggregated geometry files.")
    prepare.add_argument("--input_glob", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--config_ids", nargs="+", default=PRIMARY_COMPILER_CONFIG_IDS)
    prepare.add_argument("--allow_incomplete_smoke_test", action="store_true")
    prepare.add_argument("--causal_metric_version", choices=[CAUSAL_SCORE_VERSION], default=None)

    calibration = sub.add_parser("calibration", help="Summarize Gemma GA calibration.")
    calibration.add_argument("--results_root", default="outputs/compiler_gemma/calibration")
    calibration.add_argument("--output_dir", default="outputs/compiler_gemma/calibration")
    calibration.add_argument("--min_acquisition", type=float, default=0.5)
    calibration.add_argument("--min_transfer", type=float, default=0.5)
    calibration.add_argument("--min_boundedness", type=float, default=0.8)
    calibration.add_argument("--min_preservation", type=float, default=0.8)
    calibration.add_argument("--max_acquisition_drop", type=float, default=0.05)
    calibration.add_argument("--max_transfer_drop", type=float, default=0.05)
    calibration.add_argument("--max_failures", type=int, default=0)

    pilot = sub.add_parser("pilot", help="Build pilot geometry and headroom summaries.")
    pilot.add_argument("--input_glob", required=True)
    pilot.add_argument("--output_dir", required=True)
    pilot.add_argument("--config_ids", nargs="+", default=PRIMARY_COMPILER_CONFIG_IDS)
    pilot.add_argument("--tie_tolerance", type=float, default=1e-12)
    pilot.add_argument("--causal_metric_version", choices=[CAUSAL_SCORE_VERSION], default=None)

    compare = sub.add_parser("compare", help="Compare frozen Llama and new Gemma compiler artifacts.")
    compare.add_argument("--llama_root", default="artifacts/llama")
    compare.add_argument("--gemma_root", default="artifacts/gemma")
    compare.add_argument("--output_dir", default="outputs/release_verification/compiler_gemma/comparison")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "calibration":
        command_calibration(args)
    elif args.command == "pilot":
        command_pilot(args)
    elif args.command == "compare":
        command_compare(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
