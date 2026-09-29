#!/usr/bin/env python3
"""Evaluate one calibration-manifest row, with or without a trained adapter.

Patched for behavioral_policy scoring:
- The underlying evaluator still runs normally.
- After evaluation, behavioral_policy rows are rescored so strict accuracy is
  based on the predicted policy-action label.
- In particular, negative controls are strict-correct when the predicted label is
  NO_POLICY_TRIGGER.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import read_jsonl, load_examples_by_id
import evaluate_model


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


def _normalize_for_label_search(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    # Preserve underscores while normalizing common variants such as "ask for time".
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
    """Return recognized policy labels in order of first appearance."""
    raw = "" if text is None else str(text)
    raw_upper = raw.upper()

    hits = []
    for label in POLICY_LABELS:
        positions = []
        for variant in _label_variants(label):
            idx = raw_upper.find(variant.upper())
            if idx >= 0:
                positions.append(idx)

        # Also support whitespace-normalized variants.
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
    """Infer the gold behavioral-policy label from target/scoring fields."""
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
        ])
        for key in ["required_concepts", "required_labels", "allowed_labels"]:
            vals = scoring.get(key)
            if isinstance(vals, list):
                candidates.extend(vals)
            elif vals is not None:
                candidates.append(vals)

    # Negative-control behavioral-policy rows should target no trigger, even if
    # a legacy scoring field is incomplete.
    split = str(row.get("split", "")).lower()
    mode = str(row.get("example_mode") or row.get("metadata", {}).get("example_mode", "")).lower()
    if split == "negative_control" or mode in {"negative_control", "no_policy_trigger"}:
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


def is_behavioral_policy_row(row: Dict[str, Any], run_learning_type: Optional[str] = None) -> bool:
    if run_learning_type == "behavioral_policy":
        return True
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


def rescore_behavioral_policy_row(
    row: Dict[str, Any],
    run_learning_type: Optional[str] = None,
) -> Dict[str, Any]:
    if not is_behavioral_policy_row(row, run_learning_type=run_learning_type):
        return row

    expected = expected_policy_label(row)
    if expected is None:
        return row

    pred_text = prediction_text(row)
    pred_labels = extract_policy_labels(pred_text)

    expected_present = expected in pred_labels
    wrong_labels = [x for x in pred_labels if x != expected]

    # Strict policy-action accuracy:
    # the model should emit the expected label and no competing policy label.
    # This allows explanations like "ASK_FOR_TIME because..." but rejects
    # multi-label or wrong-label outputs.
    strict = int(expected_present and not wrong_labels)

    # Loose/concept accuracy keeps the old "target label appears somewhere" signal.
    loose = int(expected_present)
    concept = int(expected_present)

    row["behavioral_expected_label"] = expected
    row["behavioral_predicted_labels"] = pred_labels
    row["behavioral_predicted_label"] = pred_labels[0] if pred_labels else None

    # Update common metric fields used by the summarizers.
    for key in ["strict_accuracy", "strict", "is_correct", "correct"]:
        if key in row:
            row[key] = strict
    if "strict_accuracy" not in row:
        row["strict_accuracy"] = strict

    for key in ["loose_accuracy", "loose"]:
        if key in row:
            row[key] = loose
    if "loose_accuracy" not in row:
        row["loose_accuracy"] = loose

    for key in ["concept_accuracy", "concept_correct"]:
        if key in row:
            row[key] = concept
    if "concept_accuracy" not in row:
        row["concept_accuracy"] = concept

    contains = int(expected_present)
    for key in ["contains_exact", "contains_exact_rate", "target_mentioned", "target_mentioned_rate"]:
        if key in row:
            row[key] = contains

    return row


def rescore_behavioral_policy_jsonl(
    path: str | Path,
    run_learning_type: Optional[str] = None,
) -> Dict[str, int]:
    path = Path(path)
    if not path.exists():
        return {"rows": 0, "rescored": 0}

    rows = []
    total = 0
    rescored = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            before = (
                row.get("strict_accuracy"),
                row.get("loose_accuracy"),
                row.get("concept_accuracy"),
            )
            new_row = rescore_behavioral_policy_row(row, run_learning_type=run_learning_type)
            after = (
                new_row.get("strict_accuracy"),
                new_row.get("loose_accuracy"),
                new_row.get("concept_accuracy"),
            )
            if is_behavioral_policy_row(new_row, run_learning_type=run_learning_type):
                rescored += 1
            rows.append(new_row)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"rows": total, "rescored": rescored}


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
        "--model_name",
        args.model_name,
        "--examples_path",
        tmp_path,
        "--output_path",
        args.output_path,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--device_map",
        args.device_map,
        "--torch_dtype",
        args.torch_dtype,
    ]
    if args.adapter_path:
        sys.argv.extend(["--adapter_path", args.adapter_path])

    try:
        evaluate_model.main()
    finally:
        sys.argv = old_argv
        Path(tmp_path).unlink(missing_ok=True)

    # Patch scoring after the underlying evaluator writes the JSONL.
    if run.get("learning_type") == "behavioral_policy":
        stats = rescore_behavioral_policy_jsonl(
            args.output_path,
            run_learning_type=run.get("learning_type"),
        )
        print(
            f"Behavioral-policy scorer patch: rescored {stats['rescored']} "
            f"of {stats['rows']} rows in {args.output_path}"
        )


if __name__ == "__main__":
    main()
