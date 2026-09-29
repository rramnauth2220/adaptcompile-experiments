#!/usr/bin/env python3
"""Append calibration-manifest rows for behavioral_policy without modifying existing rows."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import random


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(base_seed: int, text: str) -> int:
    h = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    return base_seed + h % 100000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--learning_type", default="behavioral_policy")
    p.add_argument("--n_specs", nargs="+", type=int, default=[10, 25, 50, 100])
    p.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 12])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true", help="Replace existing rows for this learning type.")
    args = p.parse_args()

    examples = read_jsonl(args.examples_path)
    manifest_path = Path(args.manifest_path)
    manifest = read_jsonl(manifest_path) if manifest_path.exists() else []

    if args.overwrite:
        manifest = [r for r in manifest if r.get("learning_type") != args.learning_type]

    existing_run_ids = {r["run_id"] for r in manifest}

    by_spec_split = defaultdict(lambda: defaultdict(list))
    spec_ids = set()
    for e in examples:
        if e["learning_type"] != args.learning_type:
            continue
        sid = e["spec_id"]
        spec_ids.add(sid)
        by_spec_split[sid][e["split"]].append(e)

    all_sids = sorted(spec_ids)
    rng = random.Random(stable_seed(args.seed, args.learning_type))
    rng.shuffle(all_sids)

    added = 0
    for n_specs in args.n_specs:
        if n_specs > len(all_sids):
            continue
        selected_specs = sorted(all_sids[:n_specs])
        for budget in args.budgets:
            run_id = f"calib::{args.learning_type}::specs{n_specs}::budget{budget}"
            if run_id in existing_run_ids:
                continue

            train_ids, id_ids, para_ids, gen_ids, neg_ids = [], [], [], [], []
            for sid in selected_specs:
                train = sorted(
                    by_spec_split[sid]["train"],
                    key=lambda x: x.get("metadata", {}).get("train_order", 999999),
                )[:budget]
                train_ids.extend([e["example_id"] for e in train])
                id_ids.extend([e["example_id"] for e in by_spec_split[sid]["id_eval"]])
                para_ids.extend([e["example_id"] for e in by_spec_split[sid]["paraphrase_eval"]])
                gen_ids.extend([e["example_id"] for e in by_spec_split[sid]["generalization"]])
                neg_ids.extend([e["example_id"] for e in by_spec_split[sid]["negative_control"]])

            manifest.append({
                "run_id": run_id,
                "learning_type": args.learning_type,
                "n_specs": n_specs,
                "train_budget_per_spec": budget,
                "spec_ids": selected_specs,
                "train_example_ids": train_ids,
                "id_eval_example_ids": id_ids,
                "paraphrase_eval_example_ids": para_ids,
                "generalization_example_ids": gen_ids,
                "negative_control_example_ids": neg_ids,
            })
            added += 1

    write_jsonl(manifest_path, manifest)
    print(f"Added {added} manifest rows for {args.learning_type}; total rows now {len(manifest)}")


if __name__ == "__main__":
    main()
