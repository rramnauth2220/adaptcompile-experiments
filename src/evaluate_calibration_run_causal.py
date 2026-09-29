#!/usr/bin/env python3
"""Causal-mapping evaluator wrapper.

Runs the normal evaluator, then rescored causal_mapping rows using outcome-label
strict accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import read_jsonl, load_examples_by_id
import evaluate_model


LABEL_RE = re.compile(r"\b(?:EFFECT_[A-Z0-9_]+|NO_CAUSAL_EFFECT)\b")


def extract_labels(text: Any) -> List[str]:
    raw = "" if text is None else str(text).upper()
    out = []
    seen = set()
    for m in LABEL_RE.finditer(raw):
        label = m.group(0)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def expected_label(row: Dict[str, Any]) -> Optional[str]:
    candidates = [row.get("target"), row.get("exact_answer")]
    scoring = row.get("scoring")
    if isinstance(scoring, dict):
        candidates.extend([scoring.get("exact_answer"), *(scoring.get("required_concepts") or [])])
    for c in candidates:
        labels = extract_labels(c)
        if labels:
            return labels[0]
    return None


def prediction_text(row: Dict[str, Any]) -> str:
    for key in ["prediction", "generated_text", "generation", "model_output", "response", "completion", "output"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def is_causal_row(row: Dict[str, Any]) -> bool:
    if row.get("learning_type") == "causal_mapping":
        return True
    scoring = row.get("scoring")
    st = ""
    if isinstance(scoring, dict):
        st = str(scoring.get("scoring_type", ""))
    st = st or str(row.get("scoring_type", ""))
    return st.startswith("causal_") or expected_label(row) is not None


def set_metric(row: Dict[str, Any], prefix: str, value: int) -> None:
    value = int(value)
    row[prefix] = value
    row[f"{prefix}_correct"] = value
    row[f"{prefix}_accuracy"] = float(value)
    row[f"{prefix}_rate"] = float(value)


def rescore_file(path: Path) -> Dict[str, int]:
    rows = []
    total = 0
    causal = 0
    strict_sum = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1

            if is_causal_row(row):
                causal += 1
                gold = expected_label(row)
                pred_labels = extract_labels(prediction_text(row))
                wrong = [x for x in pred_labels if x != gold]
                strict = int(gold is not None and gold in pred_labels and not wrong)
                loose = int(gold is not None and gold in pred_labels)

                row["causal_expected_label"] = gold
                row["causal_predicted_labels"] = pred_labels
                row["causal_predicted_label"] = pred_labels[0] if pred_labels else None

                set_metric(row, "strict", strict)
                set_metric(row, "loose", loose)
                set_metric(row, "concept", loose)
                row["is_correct"] = strict
                row["correct"] = strict
                strict_sum += strict

            rows.append(row)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"rows": total, "causal": causal, "strict_sum": strict_sum}


def parse_args() -> argparse.Namespace:
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


def main() -> None:
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

    if run.get("learning_type") == "causal_mapping":
        stats = rescore_file(Path(args.output_path))
        print(f"Causal-mapping scorer: causal={stats['causal']}/{stats['rows']} strict_sum={stats['strict_sum']}")


if __name__ == "__main__":
    main()
