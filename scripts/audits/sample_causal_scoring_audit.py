#!/usr/bin/env python3
"""Sample blinded causal-scoring audit rows and score human labels."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from causal_content_scoring import (  # noqa: E402
    expected_label,
    score_causal_content,
    scoring_type,
    validate_alias_coverage,
)
from common import load_examples_by_id, read_jsonl  # noqa: E402
from compiler_common import read_csv, write_csv  # noqa: E402


RESPONSE_KEYS = ["response", "prediction", "generated_text", "generation", "model_output", "completion", "output"]


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(path for path in paths if path.exists()))


def response_text(row: Dict[str, Any]) -> str:
    for key in RESPONSE_KEYS:
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def load_metadata(path: Path) -> Dict[str, Any]:
    candidate = path.with_name("compiler_job_metadata.json")
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def collect_candidates(paths: Sequence[Path], examples_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        metadata = load_metadata(path)
        for row in read_jsonl(path):
            example = examples_by_id.get(str(row.get("example_id", "")))
            if not example or example.get("learning_type") != "causal_mapping":
                continue
            score = score_causal_content(example, response_text(row))
            rows.append(
                {
                    "source_path": str(path),
                    "example_id": row.get("example_id"),
                    "prompt": row.get("prompt") or example.get("prompt"),
                    "response": response_text(row),
                    "expected_label": expected_label(example),
                    "scoring_type": scoring_type(example),
                    "model_name": row.get("model_name") or metadata.get("model_name"),
                    "model_slug": row.get("model_slug") or metadata.get("model_slug"),
                    "config_id": row.get("config_id") or metadata.get("config_id"),
                    "split": row.get("split"),
                    "seed": row.get("seed") or row.get("compiler_seed") or metadata.get("seed"),
                    **score.as_fields(),
                }
            )
    return rows


def stratified_sample(rows: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row.get("scoring_type") or ""),
                str(row.get("model_slug") or "model"),
                str(row.get("config_id") or "config"),
            )
        ].append(dict(row))
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: List[Dict[str, Any]] = []
    keys = sorted(buckets)
    while keys and len(selected) < n:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < n:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    rng.shuffle(selected)
    return selected[:n]


def write_sample(sample: Sequence[Dict[str, Any]], output_csv: Path, key_csv: Path) -> None:
    blinded_rows = []
    key_rows = []
    for index, row in enumerate(sample, start=1):
        audit_id = f"causal_audit_{index:04d}"
        blinded_rows.append(
            {
                "audit_id": audit_id,
                "prompt": row["prompt"],
                "response": row["response"],
                "expected_label": row["expected_label"],
                "scoring_type": row["scoring_type"],
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "source_path": row.get("source_path"),
                "example_id": row.get("example_id"),
                "model_name": row.get("model_name"),
                "model_slug": row.get("model_slug"),
                "config_id": row.get("config_id"),
                "split": row.get("split"),
                "seed": row.get("seed"),
                "expected_label": row.get("expected_label"),
                "scoring_type": row.get("scoring_type"),
                "causal_content_correct": row.get("causal_content_correct"),
                "causal_format_correct": row.get("causal_format_correct"),
                "causal_ambiguous": row.get("causal_ambiguous"),
                "causal_score_version": row.get("causal_score_version"),
            }
        )

    write_csv(output_csv, blinded_rows, ["audit_id", "prompt", "response", "expected_label", "scoring_type"])
    write_csv(
        key_csv,
        key_rows,
        [
            "audit_id",
            "source_path",
            "example_id",
            "model_name",
            "model_slug",
            "config_id",
            "split",
            "seed",
            "expected_label",
            "scoring_type",
            "causal_content_correct",
            "causal_format_correct",
            "causal_ambiguous",
            "causal_score_version",
        ],
    )


def parse_human_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "correct"}:
        return True
    if text in {"0", "false", "f", "no", "n", "incorrect"}:
        return False
    return None


def score_labeled_audit(audit_csv: Path, key_csv: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audit_rows = read_csv(audit_csv)
    key_rows = {row["audit_id"]: row for row in read_csv(key_csv)}
    detail: List[Dict[str, Any]] = []
    for row in audit_rows:
        audit_id = row["audit_id"]
        key = key_rows.get(audit_id)
        if not key:
            continue
        human = (
            parse_human_bool(row.get("human_content_correct"))
            if "human_content_correct" in row
            else parse_human_bool(row.get("human_correct", row.get("label")))
        )
        if human is None:
            continue
        scorer = parse_human_bool(key.get("causal_content_correct"))
        detail.append(
            {
                "audit_id": audit_id,
                "scoring_type": key.get("scoring_type"),
                "human_content_correct": human,
                "scorer_content_correct": scorer,
                "agreement": human == scorer,
            }
        )

    tp = sum(1 for row in detail if row["human_content_correct"] and row["scorer_content_correct"])
    fp = sum(1 for row in detail if not row["human_content_correct"] and row["scorer_content_correct"])
    fn = sum(1 for row in detail if row["human_content_correct"] and not row["scorer_content_correct"])
    tn = sum(1 for row in detail if not row["human_content_correct"] and not row["scorer_content_correct"])
    n = len(detail)
    by_type = []
    for stype in sorted({row["scoring_type"] for row in detail}):
        group = [row for row in detail if row["scoring_type"] == stype]
        by_type.append(
            {
                "scoring_type": stype,
                "n": len(group),
                "accuracy": sum(bool(row["agreement"]) for row in group) / len(group) if group else None,
            }
        )
    summary = {
        "n_labeled": n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "agreement_by_scoring_type": by_type,
    }
    return detail, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or score a blinded causal-content audit sample.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample")
    sample.add_argument("--input_glob", nargs="+", required=True)
    sample.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    sample.add_argument("--n", type=int, default=250)
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--output_csv", required=True)
    sample.add_argument("--key_csv", default=None)

    score = sub.add_parser("score")
    score.add_argument("--audit_csv", required=True)
    score.add_argument("--key_csv", required=True)
    score.add_argument("--output_json", required=True)
    score.add_argument("--detail_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sample":
        examples_by_id = load_examples_by_id(args.examples_path)
        validate_alias_coverage([row for row in examples_by_id.values() if row.get("learning_type") == "causal_mapping"])
        candidates = collect_candidates(expand_inputs(args.input_glob), examples_by_id)
        if not candidates:
            raise ValueError("No causal rows found for audit sampling.")
        output_csv = Path(args.output_csv)
        key_csv = Path(args.key_csv) if args.key_csv else output_csv.with_suffix(".key.csv")
        write_sample(stratified_sample(candidates, args.n, args.seed), output_csv, key_csv)
        print(f"Wrote blinded audit CSV: {output_csv}")
        print(f"Wrote audit key CSV: {key_csv}")
        return

    detail, summary = score_labeled_audit(Path(args.audit_csv), Path(args.key_csv))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.detail_csv:
        write_csv(
            Path(args.detail_csv),
            detail,
            ["audit_id", "scoring_type", "human_content_correct", "scorer_content_correct", "agreement"],
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
