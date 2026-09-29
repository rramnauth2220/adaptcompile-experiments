#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

LABEL_RE = re.compile(r"\b(?:OUTCOME_[A-Z0-9_]+|PROCEDURE_INCOMPLETE)\b")


def open_jsonl(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def labels(text):
    raw = "" if text is None else str(text).upper()
    out = []
    seen = set()
    for m in LABEL_RE.finditer(raw):
        lab = m.group(0)
        if lab not in seen:
            out.append(lab)
            seen.add(lab)
    return out


def pred_text(row):
    for key in ["prediction", "generated_text", "generation", "model_output", "response", "completion", "output"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def get_metric(row, name):
    for key in [f"{name}_correct", name, f"{name}_accuracy", f"{name}_rate"]:
        if key in row and row[key] is not None:
            return float(row[key])
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No input files matched")

    rows = []
    for path in files:
        by_split = defaultdict(lambda: {"n": 0, "strict": 0, "incomplete": 0})
        with open_jsonl(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("learning_type") != "procedural_reasoning":
                    continue
                split = r.get("split", "UNKNOWN")
                labs = labels(pred_text(r))
                by_split[split]["n"] += 1
                by_split[split]["strict"] += get_metric(r, "strict")
                by_split[split]["incomplete"] += int("PROCEDURE_INCOMPLETE" in labs)

        for split, c in sorted(by_split.items()):
            n = c["n"]
            rows.append({
                "source_file": Path(path).name,
                "split": split,
                "n": n,
                "strict_accuracy": c["strict"] / n if n else 0.0,
                "incomplete_prediction_rate": c["incomplete"] / n if n else 0.0,
            })

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.round(3).to_string(index=False))
    print("Wrote", args.output_csv)


if __name__ == "__main__":
    main()
