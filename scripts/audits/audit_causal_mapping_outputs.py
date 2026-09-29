#!/usr/bin/env python3
"""Audit causal_mapping model outputs for conservative NO_CAUSAL_EFFECT overuse.

Usage:
  python scripts/audits/audit_causal_mapping_outputs.py --inputs outputs/.../*.jsonl \
    --output_csv outputs/.../causal_mapping_output_audit.csv
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

LABEL_RE = re.compile(r"\b(?:EFFECT_[A-Z0-9_]+|NO_CAUSAL_EFFECT)\b")
POSITIVE_SPLITS = {
    "id",
    "id_eval",
    "in_distribution",
    "indistribution",
    "paraphrase",
    "paraphrase_eval",
    "para",
    "generalization",
    "gen",
}


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def rows_from(path: Path) -> Iterable[Dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["source_file"] = str(path)
                yield r


def extract_labels(text: Any) -> List[str]:
    raw = "" if text is None else str(text).upper()
    out, seen = [], set()
    for m in LABEL_RE.finditer(raw):
        label = m.group(0)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def prediction_text(row: Dict[str, Any]) -> str:
    causal_labels = row.get("causal_predicted_labels")
    if isinstance(causal_labels, list) and causal_labels:
        return " ".join(str(label) for label in causal_labels)

    for key in ["prediction", "generated_text", "generation", "model_output", "response", "completion", "output"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def is_causal_row(row: Dict[str, Any]) -> bool:
    if row.get("learning_type") == "causal_mapping":
        return True
    scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}
    scoring_type = str(scoring.get("scoring_type") or row.get("scoring_type") or "")
    return scoring_type.startswith("causal_") or bool(row.get("causal_expected_label"))


def resolve_inputs(inputs: List[str]) -> List[Path]:
    files = []
    for inp in inputs:
        if any(ch in inp for ch in "*?["):
            files.extend(Path(p) for p in glob.glob(inp, recursive=True))
        else:
            files.append(Path(inp))
    return sorted({path for path in files if path.exists() and path.is_file()})


def parse_metadata_from_path(path: Path) -> Dict[str, str]:
    parts = path.parts
    text = str(path)
    meta = {
        "model_slug": "",
        "objective": "causal_mapping" if "causal_mapping" in parts or "causal_mapping" in path.name else "",
        "condition": "",
        "budget": "",
        "rank": "",
        "seed": "",
    }
    if "causal_mapping" in parts:
        idx = parts.index("causal_mapping")
        if idx > 0:
            meta["model_slug"] = parts[idx - 1]
    else:
        parent_match = re.search(r"results[\\/](llama31_8b_localization_causal)_seed", text)
        if parent_match:
            meta["model_slug"] = parent_match.group(1)

    filename = path.name
    for condition in ["full", "early", "middle", "late"]:
        if re.search(rf"(?:^|_){condition}(?:_|\.jsonl|\.jsonl\.gz$)", filename):
            meta["condition"] = condition
            break
    for key in ["seed", "budget", "rank"]:
        match = re.search(rf"(?:^|_){key}(\d+)(?:_|\.jsonl|\.jsonl\.gz$)", filename)
        if match:
            meta[key] = match.group(1)
    for part in parts:
        if part.startswith("condition_"):
            meta["condition"] = meta["condition"] or part.replace("condition_", "")
        elif part.startswith("seed_"):
            meta["seed"] = meta["seed"] or part.replace("seed_", "")
        elif part.startswith("budget_"):
            meta["budget"] = meta["budget"] or part.replace("budget_", "")
        elif part.startswith("rank_"):
            meta["rank"] = meta["rank"] or part.replace("rank_", "")
    return meta


def get_metric(row: Dict[str, Any], name: str) -> float:
    for key in [f"{name}_correct", name, f"{name}_accuracy", f"{name}_rate"]:
        if key in row and row[key] is not None:
            return float(row[key])
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()

    records = []
    for path in resolve_inputs(args.inputs):
        meta = parse_metadata_from_path(path)
        for r in rows_from(path):
            if not is_causal_row(r):
                continue
            scoring = r.get("scoring") if isinstance(r.get("scoring"), dict) else {}
            gold = r.get("causal_expected_label") or scoring.get("exact_answer") or r.get("exact_answer")
            pred_labels = r.get("causal_predicted_labels") or extract_labels(prediction_text(r))
            pred = pred_labels[0] if pred_labels else None
            no_effect = "NO_CAUSAL_EFFECT" in pred_labels
            records.append({
                **meta,
                "source_file": r["source_file"],
                "split": r.get("split"),
                "example_mode": (r.get("metadata") or {}).get("example_mode") or r.get("example_mode"),
                "scoring_type": scoring.get("scoring_type") or r.get("scoring_type"),
                "control_type": scoring.get("control_type") or (r.get("metadata") or {}).get("causal_control_type"),
                "gold": gold,
                "pred": pred,
                "strict": get_metric(r, "strict"),
                "is_no_effect_pred": float(no_effect),
                "first_label_is_no_effect": float(pred == "NO_CAUSAL_EFFECT"),
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No rows parsed.")

    group_cols = ["model_slug", "objective", "condition", "budget", "rank", "seed", "source_file", "split"]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            strict_accuracy=("strict", "mean"),
            no_effect_prediction_rate=("is_no_effect_pred", "mean"),
            first_label_no_effect_rate=("first_label_is_no_effect", "mean"),
            n=("strict", "size"),
        )
    )

    # Conservative-answer warning: high NO_CAUSAL_EFFECT use outside negative controls.
    nonneg = summary[summary["split"].isin(POSITIVE_SPLITS)]
    warnings = nonneg[nonneg["no_effect_prediction_rate"] > 0.25]

    print("\nCausal output audit by file/split:")
    print(summary.round(3).to_string(index=False))

    if not warnings.empty:
        print("\nPotential conservative NO_CAUSAL_EFFECT overuse outside negative controls:")
        print(warnings.round(3).to_string(index=False))
    else:
        print("\nNo obvious NO_CAUSAL_EFFECT overuse outside negative controls.")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
