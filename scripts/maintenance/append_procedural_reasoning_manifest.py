#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--n_specs", type=int, default=25)
    p.add_argument("--budgets", nargs="+", type=int, required=True)
    p.add_argument("--in_place", action="store_true")
    args = p.parse_args()

    examples = read_jsonl(Path(args.examples_path))
    by_spec: Dict[str, Dict[str, List[str]]] = {}
    for ex in examples:
        if ex.get("learning_type") != "procedural_reasoning":
            continue
        spec_id = ex["spec_id"]
        split = ex["split"]
        by_spec.setdefault(spec_id, {}).setdefault(split, []).append(ex["example_id"])

    spec_ids = sorted(by_spec)[: args.n_specs]
    if len(spec_ids) < args.n_specs:
        raise SystemExit(f"Only found {len(spec_ids)} procedural specs, expected {args.n_specs}")

    existing = read_jsonl(Path(args.manifest_path))
    existing = [
        r for r in existing
        if not (
            r.get("learning_type") == "procedural_reasoning"
            and int(r.get("train_budget_per_spec", r.get("budget", -1))) in args.budgets
            and r.get("n_specs") == args.n_specs
        )
    ]

    new_rows = []
    for budget in args.budgets:
        train_ids = []
        id_eval_ids = []
        paraphrase_ids = []
        gen_ids = []
        neg_ids = []
        for sid in spec_ids:
            splits = by_spec[sid]
            train_ids.extend(splits["train"][:budget])
            id_eval_ids.extend(splits["id_eval"])
            paraphrase_ids.extend(splits["paraphrase_eval"])
            gen_ids.extend(splits["generalization"])
            neg_ids.extend(splits["negative_control"])

        new_rows.append({
            "run_id": f"calib::procedural_reasoning::specs{args.n_specs}::budget{budget}",
            "learning_type": "procedural_reasoning",
            "n_specs": args.n_specs,
            "train_budget_per_spec": budget,
            "spec_ids": spec_ids,
            "train_example_ids": train_ids,
            "id_eval_example_ids": id_eval_ids,
            "paraphrase_eval_example_ids": paraphrase_ids,
            "generalization_example_ids": gen_ids,
            "negative_control_example_ids": neg_ids,
        })

    rows = existing + new_rows
    out_path = Path(args.manifest_path if args.in_place else str(args.manifest_path) + ".procedural")
    write_jsonl(out_path, rows)

    print(f"Wrote {len(new_rows)} procedural manifest rows to {out_path}")
    for r in new_rows:
        print(r["run_id"], "train", len(r["train_example_ids"]), "eval", len(r["id_eval_example_ids"])+len(r["paraphrase_eval_example_ids"])+len(r["generalization_example_ids"])+len(r["negative_control_example_ids"]))


if __name__ == "__main__":
    main()
