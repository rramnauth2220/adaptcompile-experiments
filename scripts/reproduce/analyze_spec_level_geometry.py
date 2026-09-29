#!/usr/bin/env python3
"""Analyze per-spec variation in saved localization outputs.

This diagnostic uses already-saved AAAI localization JSONLs. It should not be
treated as clean episode-level training data because those adapters were trained
jointly on multiple latent specifications.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compiler_common import (  # noqa: E402
    BOUND_SOURCE,
    CONDITIONS,
    balanced_utility,
    metric_source_for,
    metric_value,
    read_jsonl,
    write_csv,
)


def expand_inputs(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matches = glob.glob(value, recursive=True)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(value))
    return sorted(set(paths))


def infer_condition(row: Dict[str, Any], path: Path) -> str:
    for key in ["localization_condition", "condition"]:
        value = row.get(key)
        if value:
            return str(value).lower()
    m = re.search(r"_(full|early|middle|late|baseline)(?:_|\.|$)", path.name)
    return m.group(1) if m else "unknown"


def infer_seed(row: Dict[str, Any], path: Path) -> str:
    for key in ["seed", "localization_seed"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    m = re.search(r"seed[_-]?(\d+)", str(path))
    return m.group(1) if m else "unknown"


def infer_model(row: Dict[str, Any], path: Path) -> Tuple[str, str]:
    model_name = row.get("model_name") or row.get("model_name_or_path")
    model_slug = row.get("model_slug")
    if model_name or model_slug:
        return str(model_name or model_slug), str(model_slug or model_name)
    if "llama31_8b" in str(path).lower():
        return "meta-llama/Llama-3.1-8B-Instruct", "llama31_8b_instruct"
    return "unknown", "unknown"


def row_group_key(row: Dict[str, Any], path: Path) -> Tuple[str, str, str, str, str, str]:
    learning_type = str(row.get("learning_type", "unknown"))
    model_name, model_slug = infer_model(row, path)
    return (
        model_name,
        model_slug,
        learning_type,
        str(row.get("spec_id", "unknown")),
        infer_seed(row, path),
        infer_condition(row, path),
    )


def split_score(row: Dict[str, Any]) -> float | None:
    learning_type = row.get("learning_type")
    split = row.get("split")
    if not learning_type or not split or learning_type not in BOUND_SOURCE:
        return None
    try:
        source = metric_source_for(str(learning_type), str(split))
    except ValueError:
        return None
    return metric_value(row, source)


def aggregate_geometry(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_files: Dict[Tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)

    for path in paths:
        if not path.exists():
            print(f"[warn] Missing input: {path}", file=sys.stderr)
            continue
        for row in read_jsonl(path):
            key = row_group_key(row, path)
            score = split_score(row)
            if score is None:
                continue
            buckets[key][str(row.get("split"))].append(float(score))
            source_files[key].add(str(path))

    rows: List[Dict[str, Any]] = []
    for key, by_split in sorted(buckets.items()):
        model_name, model_slug, learning_type, spec_id, seed, condition = key
        id_eval = mean(by_split.get("id_eval", []))
        paraphrase = mean(by_split.get("paraphrase_eval", []))
        transfer = mean(by_split.get("generalization", []))
        boundedness = mean(by_split.get("negative_control", []))
        acquisition = None
        if id_eval is not None and paraphrase is not None:
            acquisition = (id_eval + paraphrase) / 2.0
        row = {
            "model_name": model_name,
            "model_slug": model_slug,
            "learning_type": learning_type,
            "spec_id": spec_id,
            "seed": seed,
            "localization_condition": condition,
            "id_eval": id_eval,
            "paraphrase_eval": paraphrase,
            "acquisition": acquisition,
            "transfer": transfer,
            "boundedness": boundedness,
            "exploratory_balanced_utility": None,
            "n_id_eval": len(by_split.get("id_eval", [])),
            "n_paraphrase_eval": len(by_split.get("paraphrase_eval", [])),
            "n_generalization": len(by_split.get("generalization", [])),
            "n_negative_control": len(by_split.get("negative_control", [])),
            "boundedness_metric_source": BOUND_SOURCE.get(learning_type),
            "source_files": ";".join(sorted(source_files[key])),
        }
        row["exploratory_balanced_utility"] = balanced_utility(row)
        rows.append(row)
    return rows


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def best_conditions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["model_name"],
            row["model_slug"],
            row["learning_type"],
            row["spec_id"],
            row["seed"],
        )
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    metric_names = [
        "id_eval",
        "paraphrase_eval",
        "acquisition",
        "transfer",
        "boundedness",
        "exploratory_balanced_utility",
    ]
    for key, group_rows in sorted(grouped.items()):
        model_name, model_slug, learning_type, spec_id, seed = key
        for metric in metric_names:
            scored = [r for r in group_rows if r.get(metric) is not None]
            if not scored:
                continue
            best = max(scored, key=lambda r: (float(r[metric]), -CONDITIONS.index(r["localization_condition"]) if r["localization_condition"] in CONDITIONS else 0))
            out.append(
                {
                    "model_name": model_name,
                    "model_slug": model_slug,
                    "learning_type": learning_type,
                    "spec_id": spec_id,
                    "seed": seed,
                    "metric": metric,
                    "best_condition": best["localization_condition"],
                    "best_value": best[metric],
                    "n_conditions_observed": len({r["localization_condition"] for r in scored}),
                    "is_exploratory_metric": metric == "exploratory_balanced_utility",
                }
            )
    return out


def variation_summary(best_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        grouped[(row["model_slug"], row["learning_type"], row["metric"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for (model_slug, learning_type, metric), rows in sorted(grouped.items()):
        counts = Counter(row["best_condition"] for row in rows)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total if total else 0.0
            if p:
                entropy -= p * math.log2(p)
        summaries.append(
            {
                "model_slug": model_slug,
                "learning_type": learning_type,
                "metric": metric,
                "n_spec_seed_groups": total,
                "n_best_conditions": len(counts),
                "best_condition_counts": ";".join(f"{k}:{v}" for k, v in sorted(counts.items())),
                "most_common_best_condition": counts.most_common(1)[0][0] if counts else None,
                "most_common_count": counts.most_common(1)[0][1] if counts else 0,
                "within_objective_variation_present": len(counts) > 1,
                "best_condition_entropy_bits": entropy,
                "is_exploratory_metric": metric == "exploratory_balanced_utility",
            }
        )
    return summaries


PER_SPEC_FIELDS = [
    "model_name",
    "model_slug",
    "learning_type",
    "spec_id",
    "seed",
    "localization_condition",
    "id_eval",
    "paraphrase_eval",
    "acquisition",
    "transfer",
    "boundedness",
    "exploratory_balanced_utility",
    "n_id_eval",
    "n_paraphrase_eval",
    "n_generalization",
    "n_negative_control",
    "boundedness_metric_source",
    "source_files",
]


BEST_FIELDS = [
    "model_name",
    "model_slug",
    "learning_type",
    "spec_id",
    "seed",
    "metric",
    "best_condition",
    "best_value",
    "n_conditions_observed",
    "is_exploratory_metric",
]


SUMMARY_FIELDS = [
    "model_slug",
    "learning_type",
    "metric",
    "n_spec_seed_groups",
    "n_best_conditions",
    "best_condition_counts",
    "most_common_best_condition",
    "most_common_count",
    "within_objective_variation_present",
    "best_condition_entropy_bits",
    "is_exploratory_metric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze spec-level localization geometry.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL paths or globs.")
    parser.add_argument("--output_dir", default="outputs/compiler/spec_level_geometry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    outdir = Path(args.output_dir)
    rows = aggregate_geometry(paths)
    best = best_conditions(rows)
    summary = variation_summary(best)

    write_csv(outdir / "spec_level_geometry_long.csv", rows, PER_SPEC_FIELDS)
    write_csv(outdir / "spec_level_best_conditions.csv", best, BEST_FIELDS)
    write_csv(outdir / "spec_level_variation_summary.csv", summary, SUMMARY_FIELDS)

    print(f"Read {len(paths)} input files")
    print(f"Wrote {len(rows)} per-spec rows to {outdir / 'spec_level_geometry_long.csv'}")
    print("Reminder: these outputs are diagnostic only; adapters were trained jointly on multiple specs.")


if __name__ == "__main__":
    main()
