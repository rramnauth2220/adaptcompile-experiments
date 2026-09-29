#!/usr/bin/env python3
"""
Extend data/calibration_manifest.jsonl with additional training budgets.

Why this exists:
  The calibration manifest only contains run_ids for budgets that were generated
  by make_calibration_manifest.py. If your manifest was created with budgets
  {1, 2, 4, 8, 12}, then a run_id like

      calib::lexical_binding::specs25::budget6

  will not exist, even though the dataset has enough training examples to build it.

This script adds missing manifest entries for arbitrary budgets <= the number of
available train examples per latent spec.

Example:

  python scripts/maintenance/extend_calibration_manifest_budgets.py \
    --learning_types lexical_binding \
    --n_specs 25 \
    --budgets 6 8 10 \
    --manifest_path data/calibration_manifest.jsonl \
    --examples_path data/prompt_examples.jsonl \
    --in_place

For factual:

  python scripts/maintenance/extend_calibration_manifest_budgets.py \
    --learning_types factual_association \
    --n_specs 25 \
    --budgets 8 10 12 \
    --manifest_path data/calibration_manifest.jsonl \
    --examples_path data/prompt_examples.jsonl \
    --in_place
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


EVAL_SPLITS = [
    "id_eval",
    "paraphrase_eval",
    "generalization",
    "negative_control",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", default="data/calibration_manifest.jsonl")
    p.add_argument("--examples_path", default="data/prompt_examples.jsonl")
    p.add_argument("--output_manifest_path", default=None)
    p.add_argument("--learning_types", nargs="+", required=True)
    p.add_argument("--n_specs", type=int, required=True)
    p.add_argument("--budgets", nargs="+", type=int, required=True)
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing entries for the requested run_ids. Default: keep existing entries.",
    )
    p.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite manifest_path after writing a backup.",
    )
    return p.parse_args()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_id(learning_type: str, n_specs: int, budget: int) -> str:
    return f"calib::{learning_type}::specs{n_specs}::budget{budget}"


def train_sort_key(row: Dict[str, Any]):
    meta = row.get("metadata", {})
    # Preferred: explicit mode-aware train order.
    if isinstance(meta, dict) and meta.get("train_order") is not None:
        return (0, int(meta["train_order"]), row.get("example_id", ""))

    # Fallback: prompt template id often ends with ::0, ::1, ...
    tmpl = meta.get("prompt_template_id", "") if isinstance(meta, dict) else ""
    maybe_int = None
    try:
        maybe_int = int(str(tmpl).split("::")[-1])
    except Exception:
        maybe_int = None
    if maybe_int is not None:
        return (1, maybe_int, row.get("example_id", ""))

    # Last resort: stable deterministic order.
    return (2, row.get("example_id", ""))


def build_example_index(examples: List[Dict[str, Any]]):
    by_learning_spec_split = defaultdict(list)

    for row in examples:
        learning_type = row.get("learning_type")
        spec_id = row.get("spec_id")
        split = row.get("split")
        if not learning_type or not spec_id or not split:
            continue
        by_learning_spec_split[(learning_type, spec_id, split)].append(row)

    # Stable ordering.
    for key, rows in by_learning_spec_split.items():
        if key[2] == "train":
            rows.sort(key=train_sort_key)
        else:
            rows.sort(key=lambda r: r.get("example_id", ""))

    return by_learning_spec_split


def find_template_spec_ids(
    existing_rows: List[Dict[str, Any]],
    learning_type: str,
    n_specs: int,
) -> List[str]:
    candidates = [
        row for row in existing_rows
        if row.get("learning_type") == learning_type and int(row.get("n_specs", -1)) == n_specs
    ]

    if not candidates:
        raise ValueError(
            f"No existing manifest row found for learning_type={learning_type}, n_specs={n_specs}. "
            "Run make_calibration_manifest.py first with this n_specs, then rerun this extender."
        )

    # Prefer the entry with largest existing budget; it should have the same spec_ids.
    candidates.sort(key=lambda r: int(r.get("train_budget_per_spec", 0)), reverse=True)
    spec_ids = candidates[0].get("spec_ids", [])

    if len(spec_ids) != n_specs:
        raise ValueError(
            f"Existing manifest row for {learning_type}/specs{n_specs} has {len(spec_ids)} spec_ids, "
            f"expected {n_specs}."
        )

    return spec_ids


def make_manifest_row(
    learning_type: str,
    n_specs: int,
    budget: int,
    spec_ids: List[str],
    index,
) -> Dict[str, Any]:
    train_example_ids = []

    for spec_id in spec_ids:
        train_rows = index.get((learning_type, spec_id, "train"), [])
        if len(train_rows) < budget:
            raise ValueError(
                f"Spec {spec_id} has only {len(train_rows)} train examples, "
                f"cannot create budget {budget}."
            )
        train_example_ids.extend([r["example_id"] for r in train_rows[:budget]])

    eval_ids = {}
    for split in EVAL_SPLITS:
        ids = []
        for spec_id in spec_ids:
            split_rows = index.get((learning_type, spec_id, split), [])
            ids.extend([r["example_id"] for r in split_rows])
        eval_ids[f"{split}_example_ids"] = ids

    return {
        "run_id": run_id(learning_type, n_specs, budget),
        "learning_type": learning_type,
        "n_specs": n_specs,
        "train_budget_per_spec": budget,
        "spec_ids": spec_ids,
        "train_example_ids": train_example_ids,
        **eval_ids,
        "notes": {
            "created_by": "extend_calibration_manifest_budgets.py",
            "purpose": "additional candidate budget for multi-seed confirmation sweep",
        },
    }


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    examples_path = Path(args.examples_path)

    if args.in_place:
        output_path = manifest_path
    elif args.output_manifest_path:
        output_path = Path(args.output_manifest_path)
    else:
        raise SystemExit("Specify either --in_place or --output_manifest_path.")

    existing_rows = read_jsonl(manifest_path)
    examples = read_jsonl(examples_path)
    index = build_example_index(examples)

    existing_by_run_id = {row.get("run_id"): row for row in existing_rows}
    new_rows = []
    created = []
    skipped = []
    replaced = []

    requested_run_ids = {
        run_id(lt, args.n_specs, b)
        for lt in args.learning_types
        for b in args.budgets
    }

    # Preserve existing rows, except replaced requested run_ids.
    for row in existing_rows:
        rid = row.get("run_id")
        if rid in requested_run_ids and args.replace:
            replaced.append(rid)
            continue
        new_rows.append(row)

    for learning_type in args.learning_types:
        spec_ids = find_template_spec_ids(existing_rows, learning_type, args.n_specs)

        for budget in args.budgets:
            rid = run_id(learning_type, args.n_specs, budget)

            if rid in existing_by_run_id and not args.replace:
                skipped.append(rid)
                continue

            new_rows.append(
                make_manifest_row(
                    learning_type=learning_type,
                    n_specs=args.n_specs,
                    budget=budget,
                    spec_ids=spec_ids,
                    index=index,
                )
            )
            created.append(rid)

    # Sort for readability.
    def sort_key(row):
        return (
            str(row.get("learning_type", "")),
            int(row.get("n_specs", 0)),
            int(row.get("train_budget_per_spec", 0)),
            str(row.get("run_id", "")),
        )

    new_rows.sort(key=sort_key)

    if output_path == manifest_path:
        backup = manifest_path.with_suffix(manifest_path.suffix + ".bak_budget_extend")
        shutil.copy2(manifest_path, backup)
        print(f"Wrote backup: {backup}")

    write_jsonl(output_path, new_rows)

    print(f"Wrote manifest: {output_path}")
    print(f"Created {len(created)} entries.")
    for rid in created:
        print(f"  + {rid}")

    if skipped:
        print(f"Skipped {len(skipped)} existing entries. Use --replace to overwrite.")
        for rid in skipped:
            print(f"  = {rid}")

    if replaced:
        print(f"Replaced {len(replaced)} entries.")
        for rid in replaced:
            print(f"  ~ {rid}")


if __name__ == "__main__":
    main()
