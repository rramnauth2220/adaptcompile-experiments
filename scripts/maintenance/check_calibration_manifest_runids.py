#!/usr/bin/env python3
"""
Check whether expected calibration run_ids exist in the manifest.

Example:
  python scripts/maintenance/check_calibration_manifest_runids.py \
    --learning_type lexical_binding \
    --n_specs 25 \
    --budgets 6 8 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--learning_type", required=True)
    p.add_argument("--n_specs", type=int, required=True)
    p.add_argument("--budgets", nargs="+", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)

    run_ids = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                run_ids.add(json.loads(line)["run_id"])

    missing = []
    for budget in args.budgets:
        rid = f"calib::{args.learning_type}::specs{args.n_specs}::budget{budget}"
        if rid in run_ids:
            print(f"OK      {rid}")
        else:
            print(f"MISSING {rid}")
            missing.append(rid)

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
