#!/usr/bin/env python3
"""Audit behavioral_policy examples for counts, modes, and action labels."""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    args = p.parse_args()

    rows = [r for r in read_jsonl(args.examples_path) if r["learning_type"] == "behavioral_policy"]
    by_split = Counter(r["split"] for r in rows)
    by_mode = Counter(r.get("metadata", {}).get("example_mode") for r in rows)
    by_action = Counter(r.get("latent_spec", {}).get("action_label") for r in rows)
    by_scoring = Counter(r.get("scoring", {}).get("scoring_type") for r in rows)

    print(f"behavioral_policy examples: {len(rows)}")
    print("split counts:")
    for k, v in sorted(by_split.items()): print(f"  {k}: {v}")
    print("mode counts:")
    for k, v in sorted(by_mode.items()): print(f"  {k}: {v}")
    print("action counts:")
    for k, v in sorted(by_action.items()): print(f"  {k}: {v}")
    print("scoring counts:")
    for k, v in sorted(by_scoring.items()): print(f"  {k}: {v}")

    assert by_split == {"train": 1440, "id_eval": 600, "paraphrase_eval": 600, "generalization": 720, "negative_control": 720}, by_split
    assert len(by_action) >= 8, by_action
    assert "NO_POLICY_TRIGGER" in {r["scoring"].get("exact_answer") for r in rows if r["split"] == "negative_control"}
    print("OK: behavioral_policy dataset audit passed.")


if __name__ == "__main__":
    main()
