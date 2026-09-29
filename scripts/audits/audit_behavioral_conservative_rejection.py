#!/usr/bin/env python3
"""Audit behavioral-policy conservative NO_POLICY_TRIGGER predictions.

Behavioral boundedness is measured on negative controls whose correct label is
NO_POLICY_TRIGGER. This audit checks whether high boundedness is inflated by
conservative collapse: predicting NO_POLICY_TRIGGER on positive behavioral
examples from id_eval, paraphrase_eval, and generalization.
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


def normalize_for_label_search(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.upper()
    text = re.sub(r"[\-\/]+", "_", text)
    text = re.sub(r"[^A-Z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_variants(label: str) -> list[str]:
    return [
        label,
        label.replace("_", " "),
        label.replace("_", "-"),
        label.lower(),
        label.lower().replace("_", " "),
        label.lower().replace("_", "-"),
    ]


def extract_policy_labels(text: Any) -> list[str]:
    raw = "" if text is None else str(text)
    raw_upper = raw.upper()
    normed = normalize_for_label_search(raw)

    hits = []
    for label in POLICY_LABELS:
        positions = []
        for variant in label_variants(label):
            idx = raw_upper.find(variant.upper())
            if idx >= 0:
                positions.append(idx)

        norm_label_text = label.replace("_", " ")
        idx2 = normed.find(norm_label_text)
        if idx2 >= 0:
            positions.append(idx2)

        if positions:
            hits.append((min(positions), label))

    hits.sort(key=lambda item: item[0])
    out = []
    seen = set()
    for _pos, label in hits:
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
    split = get_first(row, ["split", "eval_split", "dataset_split", "subset", "mode", "eval_mode"])
    return norm_label(split).lower()


def get_prediction(row: Dict[str, Any]) -> str:
    labels = row.get("behavioral_predicted_labels")
    if isinstance(labels, list) and labels:
        return " ".join(norm(x) for x in labels)

    return get_first(
        row,
        [
            "behavioral_predicted_label",
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
            "behavioral_expected_label",
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


def is_behavioral_file(path: Path) -> bool:
    return "behavioral_policy" in str(path)


def is_result_file(path: Path) -> bool:
    name = path.name.lower()
    return (name.endswith(".jsonl") or name.endswith(".jsonl.gz")) and "train" not in name


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
        "objective": "behavioral_policy" if "behavioral_policy" in parts else "",
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
    if "behavioral_policy" in parts:
        idx = parts.index("behavioral_policy")
        if idx > 0:
            meta["model_slug"] = parts[idx - 1]
    else:
        parent_match = re.search(r"results[\\/](llama31_8b_localization_behavioral)_seed", text)
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


def prediction_is_no_trigger(prediction: str) -> bool:
    pred_norm = norm_label(prediction)
    return pred_norm == "NO_POLICY_TRIGGER" or "NO_POLICY_TRIGGER" in extract_policy_labels(prediction)


def target_is_no_trigger(target: str) -> bool:
    target_norm = norm_label(target)
    return target_norm == "NO_POLICY_TRIGGER" or "NO_POLICY_TRIGGER" in extract_policy_labels(target)


def split_is_positive(split: str) -> bool:
    return split in POSITIVE_SPLITS


def audit_file(path: Path) -> Optional[Dict[str, Any]]:
    meta = parse_metadata_from_path(path)
    total_positive = 0
    null_on_positive = 0
    by_split = defaultdict(lambda: {"n": 0, "null": 0})

    for row in read_jsonl(path):
        if row.get("learning_type") not in (None, "", "behavioral_policy"):
            continue
        split = get_split(row)
        if not split_is_positive(split):
            continue

        target = get_target(row)
        if target and target_is_no_trigger(target):
            continue

        pred = get_prediction(row)
        total_positive += 1
        by_split[split]["n"] += 1
        if prediction_is_no_trigger(pred):
            null_on_positive += 1
            by_split[split]["null"] += 1

    if total_positive == 0:
        return None

    out = {
        **meta,
        "result_file": str(path),
        "positive_examples": total_positive,
        "no_policy_trigger_on_positive": null_on_positive,
        "rate": null_on_positive / total_positive,
    }
    for split in sorted(by_split):
        out[f"{split}_n"] = by_split[split]["n"]
        out[f"{split}_no_policy_trigger"] = by_split[split]["null"]
        out[f"{split}_rate"] = by_split[split]["null"] / by_split[split]["n"] if by_split[split]["n"] else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"), help="Root directory to scan.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Optional explicit JSONL/JSONL.GZ files or glob patterns. Overrides --root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/behavioral_conservative_rejection_audit.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.inputs:
        files = [path for path in resolve_inputs(args.inputs) if is_result_file(path) and is_behavioral_file(path)]
    else:
        files = [
            path
            for path in args.root.rglob("*")
            if path.is_file() and is_result_file(path) and is_behavioral_file(path)
        ]

    rows = [row for path in sorted(files) if (row := audit_file(path)) is not None]
    if not rows:
        raise SystemExit(
            f"No behavioral audit rows found under {args.root}. "
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
        "no_policy_trigger_on_positive",
        "rate",
        "result_file",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_n = sum(int(row["positive_examples"]) for row in rows)
    total_null = sum(int(row["no_policy_trigger_on_positive"]) for row in rows)
    print(f"Wrote: {args.output}")
    print(f"Audited files: {len(rows)}")
    print(f"Positive behavioral examples: {total_n}")
    print(f"NO_POLICY_TRIGGER on positives: {total_null}")
    print(f"Rate: {total_null / total_n if total_n else 0:.6f}")

    print("\nBy condition:")
    cond = defaultdict(lambda: {"n": 0, "null": 0})
    for row in rows:
        condition = row.get("condition", "") or "unknown"
        cond[condition]["n"] += int(row["positive_examples"])
        cond[condition]["null"] += int(row["no_policy_trigger_on_positive"])

    for condition in sorted(cond):
        n = cond[condition]["n"]
        k = cond[condition]["null"]
        print(f"  {condition:>8}: {k}/{n} = {k / n if n else 0:.6f}")


if __name__ == "__main__":
    main()
