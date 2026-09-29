#!/usr/bin/env python3
"""Behavioral-policy evaluator wrapper.

This is a separated behavioral evaluator so lexical/factual can keep using the
original `src/evaluate_calibration_run.py`.

It runs the normal evaluator, then rescored behavioral_policy rows using V2
policy-label metrics.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path

from common import read_jsonl, load_examples_by_id
import evaluate_model


import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


POLICY_LABELS = [
    "ASK_FOR_TIME",
    "ASK_FOR_RECIPIENT",
    "ASK_FOR_FORMAT",
    "ASK_FOR_BUDGET",
    "ASK_FOR_LOCATION",
    "STATE_UNCERTAINTY",
    "CONFIRM_BEFORE_ACTING",
    "COMPARE_OPTIONS",
    "ASK_FOR_SYMPTOMS",
    "ASK_FOR_PREFERENCE",
    "NO_POLICY_TRIGGER",
]


def _normalize_for_label_search(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[\-\/]+", "_", text)
    text = re.sub(r"[^A-Z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_variants(label: str) -> List[str]:
    return [
        label,
        label.replace("_", " "),
        label.replace("_", "-"),
        label.lower(),
        label.lower().replace("_", " "),
        label.lower().replace("_", "-"),
    ]


def extract_policy_labels(text: Any) -> List[str]:
    raw = "" if text is None else str(text)
    raw_upper = raw.upper()

    hits = []
    for label in POLICY_LABELS:
        positions = []
        for variant in _label_variants(label):
            idx = raw_upper.find(variant.upper())
            if idx >= 0:
                positions.append(idx)

        norm = _normalize_for_label_search(raw)
        norm_label = label.replace("_", " ")
        idx2 = norm.find(norm_label)
        if idx2 >= 0:
            positions.append(idx2)

        if positions:
            hits.append((min(positions), label))

    hits.sort(key=lambda x: x[0])

    ordered = []
    seen = set()
    for _, label in hits:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def expected_policy_label(row: Dict[str, Any]) -> Optional[str]:
    candidates: List[Any] = [
        row.get("target"),
        row.get("exact_answer"),
        row.get("gold"),
        row.get("answer"),
    ]

    scoring = row.get("scoring")
    if isinstance(scoring, dict):
        candidates.extend([
            scoring.get("exact_answer"),
            scoring.get("target_action"),
            scoring.get("policy_action"),
            scoring.get("expected_label"),
            scoring.get("label"),
        ])
        for key in ["required_concepts", "required_labels", "allowed_labels"]:
            vals = scoring.get(key)
            if isinstance(vals, list):
                candidates.extend(vals)
            elif vals is not None:
                candidates.append(vals)

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend([
            metadata.get("target_action"),
            metadata.get("policy_action"),
            metadata.get("expected_label"),
            metadata.get("label"),
        ])

    split = str(row.get("split", "")).lower()
    mode = str(row.get("example_mode") or (metadata or {}).get("example_mode", "")).lower()
    scoring_type = str(row.get("scoring_type") or (scoring or {}).get("scoring_type", "")).lower()

    if (
        split == "negative_control"
        or mode in {"negative_control", "no_policy_trigger", "policy_negative_control"}
        or scoring_type in {"behavioral_policy_no_trigger", "policy_no_trigger"}
    ):
        candidates.insert(0, "NO_POLICY_TRIGGER")

    for c in candidates:
        labels = extract_policy_labels(c)
        if labels:
            return labels[0]

    return None


def prediction_text(row: Dict[str, Any]) -> str:
    for key in [
        "prediction",
        "generated_text",
        "generation",
        "model_output",
        "response",
        "completion",
        "output",
    ]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def is_behavioral_policy_row(row: Dict[str, Any]) -> bool:
    if row.get("learning_type") == "behavioral_policy":
        return True

    scoring = row.get("scoring")
    scoring_type = ""
    if isinstance(scoring, dict):
        scoring_type = str(scoring.get("scoring_type", ""))
    scoring_type = scoring_type or str(row.get("scoring_type", ""))

    if scoring_type.startswith("behavioral_policy") or scoring_type.startswith("policy_"):
        return True

    return expected_policy_label(row) is not None


def set_metric_fields(row: Dict[str, Any], prefix: str, value: int) -> None:
    """Set every common row-level metric variant used by summarizers."""
    value = int(value)
    rate_value = float(value)

    for key in [prefix, f"{prefix}_correct", f"is_{prefix}"]:
        if key in row:
            row[key] = value

    for key in [f"{prefix}_accuracy", f"{prefix}_rate"]:
        if key in row:
            row[key] = rate_value

    row[f"{prefix}_correct"] = value
    row[f"{prefix}_accuracy"] = rate_value


def rescore_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not is_behavioral_policy_row(row):
        return row

    expected = expected_policy_label(row)
    if expected is None:
        return row

    pred_labels = extract_policy_labels(prediction_text(row))
    expected_present = expected in pred_labels
    wrong_labels = [x for x in pred_labels if x != expected]

    strict = int(expected_present and not wrong_labels)
    loose = int(expected_present)
    concept = int(expected_present)

    row["behavioral_expected_label"] = expected
    row["behavioral_predicted_labels"] = pred_labels
    row["behavioral_predicted_label"] = pred_labels[0] if pred_labels else None

    set_metric_fields(row, "strict", strict)
    set_metric_fields(row, "loose", loose)
    set_metric_fields(row, "concept", concept)
    set_metric_fields(row, "passed", strict)

    row["is_correct"] = strict
    row["correct"] = strict

    contains = int(expected_present)
    for key in [
        "contains_exact",
        "contains_exact_correct",
        "contains_exact_rate",
        "target_mentioned",
        "target_mentioned_correct",
        "target_mentioned_rate",
    ]:
        if key in row:
            row[key] = contains if not key.endswith("_rate") else float(contains)

    row["contains_exact"] = contains
    row["contains_exact_rate"] = float(contains)
    row["target_mentioned"] = contains
    row["target_mentioned_rate"] = float(contains)

    return row


def rescore_file(path: Path, output_path: Optional[Path], backup: bool = False) -> Dict[str, int]:
    rows = []
    total = 0
    behavioral = 0
    strict_sum = 0
    concept_sum = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if is_behavioral_policy_row(row):
                behavioral += 1
            row = rescore_row(row)
            strict_sum += int(row.get("strict_correct", row.get("strict_accuracy", 0)) or 0)
            concept_sum += int(row.get("concept_correct", row.get("concept_accuracy", 0)) or 0)
            rows.append(row)

    dest = output_path or path

    if output_path is None and backup:
        bak = path.with_suffix(path.suffix + ".bak_before_behavioral_rescore_v2")
        if not bak.exists():
            shutil.copy2(path, bak)

    with open(dest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "rows": total,
        "behavioral": behavioral,
        "strict_sum": strict_sum,
        "concept_sum": concept_sum,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--run_id", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--torch_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()

    examples_by_id = load_examples_by_id(args.examples_path)
    run = next((r for r in read_jsonl(args.manifest_path) if r["run_id"] == args.run_id), None)
    if run is None:
        raise ValueError(f"Could not find run_id={args.run_id}")

    eval_ids = (
        run["id_eval_example_ids"]
        + run["paraphrase_eval_example_ids"]
        + run["generalization_example_ids"]
        + run["negative_control_example_ids"]
    )
    eval_rows = [examples_by_id[eid] for eid in eval_ids]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        for row in eval_rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")

    old_argv = sys.argv
    sys.argv = [
        "evaluate_model.py",
        "--model_name", args.model_name,
        "--examples_path", tmp_path,
        "--output_path", args.output_path,
        "--max_new_tokens", str(args.max_new_tokens),
        "--device_map", args.device_map,
        "--torch_dtype", args.torch_dtype,
    ]
    if args.adapter_path:
        sys.argv.extend(["--adapter_path", args.adapter_path])

    try:
        evaluate_model.main()
    finally:
        sys.argv = old_argv
        Path(tmp_path).unlink(missing_ok=True)

    if run.get("learning_type") == "behavioral_policy":
        stats = rescore_file(Path(args.output_path), output_path=None, backup=False)
        print(
            f"Behavioral-policy V2 scorer: behavioral={stats['behavioral']}/{stats['rows']} "
            f"strict_sum={stats['strict_sum']} concept_sum={stats['concept_sum']}"
        )


if __name__ == "__main__":
    main()
