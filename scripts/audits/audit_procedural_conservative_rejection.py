#!/usr/bin/env python3
"""
Audit procedural conservative rejection.

For procedural reasoning, boundedness is measured on negative controls whose
correct label is PROCEDURE_INCOMPLETE. This audit checks whether high
boundedness is inflated by conservative collapse: predicting
PROCEDURE_INCOMPLETE even on positive procedural examples.

The script scans result JSONL/JSONL.GZ files, keeps procedural_reasoning runs,
and counts PROCEDURE_INCOMPLETE predictions on positive splits:
ID / paraphrase / generalization.

Example:
  python3 scripts/audits/audit_procedural_conservative_rejection.py \
    --root artifacts/localization/primary \
    --output outputs/release_verification/procedural_conservative_rejection_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


NULL_LABELS = {
    "PROCEDURE_INCOMPLETE",
    "procedure_incomplete",
    "Procedure incomplete",
    "procedure incomplete",
}
LABEL_RE = re.compile(r"\b(?:OUTCOME_[A-Z0-9_]+|PROCEDURE_INCOMPLETE)\b")

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


def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def norm(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def norm_label(text: Any) -> str:
    text = norm(text)
    text = text.replace("-", "_").replace(" ", "_")
    return text.upper()


def extract_labels(text: Any) -> list[str]:
    raw = "" if text is None else str(text).upper()
    out = []
    seen = set()
    for match in LABEL_RE.finditer(raw):
        label = match.group(0)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def get_first(row: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return norm(row[key])
    return ""


def get_split(row: Dict[str, Any]) -> str:
    split = get_first(
        row,
        [
            "split",
            "eval_split",
            "dataset_split",
            "subset",
            "mode",
            "eval_mode",
        ],
    )
    return norm_label(split).lower()


def get_prediction(row: Dict[str, Any]) -> str:
    procedural_labels = row.get("procedural_predicted_labels")
    if isinstance(procedural_labels, list) and procedural_labels:
        return " ".join(norm(x) for x in procedural_labels)

    return get_first(
        row,
        [
            "procedural_predicted_label",
            "parsed_prediction",
            "prediction",
            "predicted_label",
            "model_label",
            "decoded_prediction",
            "response",
            "model_output",
            "output",
            "completion",
        ],
    )


def get_target(row: Dict[str, Any]) -> str:
    return get_first(
        row,
        [
            "target",
            "gold",
            "gold_label",
            "label",
            "answer",
            "correct_answer",
            "expected",
            "expected_label",
        ],
    )


def is_procedural_file(path: Path) -> bool:
    return "procedural_reasoning" in str(path)


def is_result_file(path: Path) -> bool:
    name = path.name.lower()
    if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
        return False
    # Avoid obvious training files if they happen to be jsonl.
    bad = {"train", "training"}
    return not any(token in name for token in bad)


def resolve_inputs(inputs: list[str]) -> list[Path]:
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
        "objective": "procedural_reasoning" if "procedural_reasoning" in parts else "",
        "condition": "",
        "seed": "",
        "budget": "",
        "rank": "",
        "model_slug": "",
    }

    for part in parts:
        if part.startswith("condition_"):
            meta["condition"] = part.replace("condition_", "")
        elif part.startswith("seed_"):
            meta["seed"] = part.replace("seed_", "")
        elif part.startswith("budget_"):
            meta["budget"] = part.replace("budget_", "")
        elif part.startswith("rank_"):
            meta["rank"] = part.replace("rank_", "")

    # Heuristic: model slug is often the directory before objective.
    if "procedural_reasoning" in parts:
        idx = parts.index("procedural_reasoning")
        if idx > 0:
            meta["model_slug"] = parts[idx - 1]
    else:
        parent_match = re.search(r"results[\\/](llama31_8b_localization_procedural)_seed", text)
        if parent_match:
            meta["model_slug"] = parent_match.group(1)

    filename = path.name
    for condition in ["full", "early", "middle", "late"]:
        if re.search(rf"(?:^|_){condition}(?:_|\.jsonl|\.jsonl\.gz$)", filename):
            meta["condition"] = meta["condition"] or condition
            break
    seed_match = re.search(r"(?:^|_)seed(\d+)(?:_|\.jsonl|\.jsonl\.gz$)", filename)
    if seed_match:
        meta["seed"] = meta["seed"] or seed_match.group(1)
    budget_match = re.search(r"(?:^|_)budget(\d+)(?:_|\.jsonl|\.jsonl\.gz$)", filename)
    if budget_match:
        meta["budget"] = meta["budget"] or budget_match.group(1)
    rank_match = re.search(r"(?:^|_)rank(\d+)(?:_|\.jsonl|\.jsonl\.gz$)", filename)
    if rank_match:
        meta["rank"] = meta["rank"] or rank_match.group(1)

    return meta


def prediction_is_incomplete(prediction: str) -> bool:
    pred_norm = norm_label(prediction)
    return pred_norm == "PROCEDURE_INCOMPLETE" or "PROCEDURE_INCOMPLETE" in extract_labels(prediction)


def target_is_incomplete(target: str) -> bool:
    target_norm = norm_label(target)
    return target_norm == "PROCEDURE_INCOMPLETE" or "PROCEDURE_INCOMPLETE" in extract_labels(target)


def split_is_positive(split: str) -> bool:
    return split in POSITIVE_SPLITS


def audit_file(path: Path) -> Optional[Dict[str, Any]]:
    meta = parse_metadata_from_path(path)

    total_positive = 0
    null_on_positive = 0

    by_split = defaultdict(lambda: {"n": 0, "null": 0})

    for row in read_jsonl(path):
        split = get_split(row)
        pred = get_prediction(row)
        target = get_target(row)

        if not split_is_positive(split):
            continue

        # Only audit examples whose target is non-null.
        # If target is missing, keep the example unless the split is negative;
        # most procedural positive splits should have non-null causal/procedure labels.
        if target and target_is_incomplete(target):
            continue

        total_positive += 1
        by_split[split]["n"] += 1

        if prediction_is_incomplete(pred):
            null_on_positive += 1
            by_split[split]["null"] += 1

    if total_positive == 0:
        return None

    row = {
        **meta,
        "result_file": str(path),
        "positive_examples": total_positive,
        "procedure_incomplete_on_positive": null_on_positive,
        "rate": null_on_positive / total_positive,
    }

    for split in sorted(by_split):
        row[f"{split}_n"] = by_split[split]["n"]
        row[f"{split}_procedure_incomplete"] = by_split[split]["null"]
        row[f"{split}_rate"] = (
            by_split[split]["null"] / by_split[split]["n"]
            if by_split[split]["n"]
            else 0.0
        )

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Optional explicit JSONL/JSONL.GZ files or glob patterns. Overrides --root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/procedural_conservative_rejection_audit.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.inputs:
        files = [p for p in resolve_inputs(args.inputs) if is_result_file(p) and is_procedural_file(p)]
    else:
        files = [
            p
            for p in args.root.rglob("*")
            if p.is_file() and is_result_file(p) and is_procedural_file(p)
        ]

    rows = []
    for path in sorted(files):
        result = audit_file(path)
        if result is not None:
            rows.append(result)

    if not rows:
        raise SystemExit(
            f"No procedural audit rows found under {args.root}. "
            "Check that result JSONL files exist and contain split/prediction fields."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model_slug",
        "objective",
        "condition",
        "budget",
        "rank",
        "seed",
        "positive_examples",
        "procedure_incomplete_on_positive",
        "rate",
        "result_file",
    ]
    fieldnames = preferred + [f for f in fieldnames if f not in preferred]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_n = sum(int(r["positive_examples"]) for r in rows)
    total_null = sum(int(r["procedure_incomplete_on_positive"]) for r in rows)
    rate = total_null / total_n if total_n else 0.0

    print(f"Wrote: {args.output}")
    print(f"Audited files: {len(rows)}")
    print(f"Positive procedural examples: {total_n}")
    print(f"PROCEDURE_INCOMPLETE on positives: {total_null}")
    print(f"Rate: {rate:.6f}")

    print("\nBy condition:")
    cond = defaultdict(lambda: {"n": 0, "null": 0})
    for r in rows:
        c = r.get("condition", "") or "unknown"
        cond[c]["n"] += int(r["positive_examples"])
        cond[c]["null"] += int(r["procedure_incomplete_on_positive"])

    for c in sorted(cond):
        n = cond[c]["n"]
        k = cond[c]["null"]
        print(f"  {c:>8}: {k}/{n} = {k / n if n else 0:.6f}")


if __name__ == "__main__":
    main()
