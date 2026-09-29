#!/usr/bin/env python3
"""Append causal_mapping calibration rows to data/calibration_manifest.jsonl."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--learning_type", default="causal_mapping")
    p.add_argument("--n_specs", type=int, default=25)
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    p.add_argument("--in_place", action="store_true")
    args = p.parse_args()

    examples_path = Path(args.examples_path)
    manifest_path = Path(args.manifest_path)
    examples = [e for e in read_jsonl(examples_path) if e.get("learning_type") == args.learning_type]
    if not examples:
        raise SystemExit(f"No examples found for learning_type={args.learning_type}. Regenerate data first.")

    by_spec_split: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        by_spec_split[ex["spec_id"]][ex["split"]].append(ex)

    spec_ids = sorted(by_spec_split.keys())[: args.n_specs]
    if len(spec_ids) < args.n_specs:
        raise SystemExit(f"Requested n_specs={args.n_specs}, but only found {len(spec_ids)} specs.")

    rows = read_jsonl(manifest_path)
    new_run_ids = {f"calib::{args.learning_type}::specs{args.n_specs}::budget{b}" for b in args.budgets}
    rows = [r for r in rows if r.get("run_id") not in new_run_ids]

    for budget in args.budgets:
        if budget > 12:
            raise SystemExit("Budget exceeds available train examples per spec (12).")

        train_ids: List[str] = []
        id_eval_ids: List[str] = []
        paraphrase_ids: List[str] = []
        generalization_ids: List[str] = []
        negative_ids: List[str] = []

        for sid in spec_ids:
            splits = by_spec_split[sid]
            train = sorted(splits["train"], key=lambda x: x["metadata"].get("train_order") or 0)
            train_ids.extend([x["example_id"] for x in train[:budget]])
            id_eval_ids.extend([x["example_id"] for x in sorted(splits["id_eval"], key=lambda x: x["example_id"])])
            paraphrase_ids.extend([x["example_id"] for x in sorted(splits["paraphrase_eval"], key=lambda x: x["example_id"])])
            generalization_ids.extend([x["example_id"] for x in sorted(splits["generalization"], key=lambda x: x["example_id"])])
            negative_ids.extend([x["example_id"] for x in sorted(splits["negative_control"], key=lambda x: x["example_id"])])

        rows.append({
            "run_id": f"calib::{args.learning_type}::specs{args.n_specs}::budget{budget}",
            "learning_type": args.learning_type,
            "n_specs": args.n_specs,
            "train_budget_per_spec": budget,
            "spec_ids": spec_ids,
            "train_example_ids": train_ids,
            "id_eval_example_ids": id_eval_ids,
            "paraphrase_eval_example_ids": paraphrase_ids,
            "generalization_example_ids": generalization_ids,
            "negative_control_example_ids": negative_ids,
            "metadata": {
                "created_by": "append_causal_mapping_manifest.py",
                "selection": "first_n_specs_sorted_by_spec_id_first_budget_train_examples",
            },
        })

    out = manifest_path if args.in_place else manifest_path.with_name(manifest_path.stem + "_with_causal.jsonl")
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} manifest rows to {out}")
    for rid in sorted(new_run_ids):
        print(rid)


if __name__ == "__main__":
    main()
