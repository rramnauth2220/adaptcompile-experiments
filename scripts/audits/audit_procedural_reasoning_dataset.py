#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--tokenizer_name", default=None)
    args = p.parse_args()

    rows = []
    with open(args.examples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("learning_type") == "procedural_reasoning":
                    rows.append(r)

    print(f"Procedural examples: {len(rows)}")
    print(f"Procedural specs: {len(set(r['spec_id'] for r in rows))}")

    by_split = Counter(r["split"] for r in rows)
    print("\nSplit counts:")
    for k, v in sorted(by_split.items()):
        print(f"  {k:18s} {v}")

    by_mode = Counter(r.get("metadata", {}).get("example_mode", "UNKNOWN") for r in rows)
    print("\nMode counts:")
    for k, v in sorted(by_mode.items()):
        print(f"  {k:35s} {v}")

    labels = Counter(r.get("scoring", {}).get("exact_answer", "UNKNOWN") for r in rows)
    print("\nExact-answer label counts:")
    for k, v in labels.most_common():
        print(f"  {k:35s} {v}")

    leakage = []
    for r in rows:
        label = r.get("scoring", {}).get("exact_answer")
        prompt = r.get("prompt", "")
        mode = r.get("metadata", {}).get("example_mode")
        # Positive-polarity prompts intentionally name the candidate label.
        if label and label in prompt and mode != "procedure_positive_polarity":
            leakage.append((r["example_id"], r["split"], mode, label))
    print(f"\nTarget-label prompt leakage outside positive-polarity prompts: {len(leakage)}")
    for item in leakage[:10]:
        print("  ", item)

    control_types = Counter(r.get("scoring", {}).get("control_type") for r in rows if r["split"] == "negative_control")
    print("\nNegative-control types:")
    for k, v in sorted(control_types.items()):
        print(f"  {str(k):25s} {v}")

    if args.tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        token_counts = defaultdict(list)
        for r in rows:
            label = r.get("scoring", {}).get("exact_answer")
            if label:
                token_counts[label].append(len(tok.encode(label, add_special_tokens=False)))
        print("\nOutcome-label token lengths:")
        for label, vals in sorted(token_counts.items()):
            print(f"  {label:35s} mean={sum(vals)/len(vals):.2f} min={min(vals)} max={max(vals)}")


if __name__ == "__main__":
    main()
