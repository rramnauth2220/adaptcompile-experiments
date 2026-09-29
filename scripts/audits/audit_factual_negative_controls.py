#!/usr/bin/env python3
"""
Audit factual-association negative controls for retrieval-template collisions.

Run after regenerating data:

  python scripts/audits/audit_factual_negative_controls.py --examples_path data/prompt_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_PATTERNS = [
    re.compile(r"^\s*What is the capital of\b", re.I),
    re.compile(r"^\s*Who invented\b", re.I),
    re.compile(r"^\s*Where does\b", re.I),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--max_examples", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.examples_path)

    bad = []
    total = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            if row.get("learning_type") != "factual_association":
                continue
            if row.get("split") != "negative_control":
                continue

            total += 1
            prompt = row.get("prompt", "")
            if any(p.search(prompt) for p in BAD_PATTERNS):
                bad.append(row)

    print(f"Checked factual negative-control rows: {total}")

    if bad:
        print(f"Found {len(bad)} retrieval-like factual negative-control prompts.")
        print()
        for row in bad[: args.max_examples]:
            meta = row.get("metadata", {})
            print(f"example_id: {row.get('example_id')}")
            print(f"relation:   {meta.get('relation_type') or row.get('latent_spec', {}).get('relation')}")
            print(f"prompt:     {row.get('prompt')}")
            print()
        raise SystemExit(1)

    print("OK: no retrieval-like factual negative-control prompts found.")


if __name__ == "__main__":
    main()
